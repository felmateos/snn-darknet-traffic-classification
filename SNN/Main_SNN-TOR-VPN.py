import os
import psutil
import sys
import numpy as np
import torch
from torch.autograd.profiler import profile
import uuid
import time
import warnings
import termcolor
import seaborn as sns
import yaml
from dacite import from_dict
from dataclasses import asdict

import utils
import models
import optim
import data
from config import UserParams

sys.path.append("./")
warnings.simplefilter("ignore", category=UserWarning)

with open("best_params.yaml", "r") as file:
    params = yaml.safe_load(file)

User_params = from_dict(data_class=UserParams, data=params)
User_params = asdict(User_params)

#=======================================================================================
if (User_params['Environment']!="Server"):
    import matplotlib.pyplot as plt
else :
    User_params['Plot_graphs'] = False
    User_params['plot_show'] = False

root_dir = "."
root_dir_data = root_dir + "/Dataset"

Save_path_batch = root_dir + "/Model/Model_batch.pth"
Save_path_epoch = root_dir + "/Model/Model_epoch.pth"
Save_path_final = root_dir + "/Model/"
Load_path = root_dir + User_params['load_path']
Test_path = root_dir + "/Model_test/"
Results_path = root_dir + User_params['Results_path']
Find_path = root_dir + User_params['find_params']['Find_path']

use_cuda = False #(User_params['Environment'] == "Server") or (User_params['Environment'] == "colab")
# =================================================================================

dtype = torch.float
np_dtype = float # np.float
# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("number of GPUs available : ",torch.cuda.device_count())
    print("cuda capability of the device : ",torch.cuda.get_device_capability(device))
    print("Name of the device : ",torch.cuda.get_device_name(device))
    # print("torch memory_stats : ", torch.cuda.memory_stats(device))
    print("torch memory_summary : ", torch.cuda.memory_summary(device))
    print("torch memory_allocated : ", torch.cuda.memory_allocated(device))
    print("torch max_memory_allocated : ", torch.cuda.max_memory_allocated(device))
    print("torch memory_reserved : ", torch.cuda.memory_reserved(device))
    print("torch max_memory_reserved : ", torch.cuda.max_memory_reserved(device))
    print("torch memory_cached : ", torch.cuda.memory_cached(device))
    print("torch max_memory_cached : ", torch.cuda.max_memory_cached(device))
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device)
    torch.cuda.reset_max_memory_cached(device)
    User_params['device']['device_type'] = str(device)
    User_params['device']['device_name'] = str(torch.cuda.get_device_name(device))
    User_params['device']['device_capability'] = str(torch.cuda.get_device_capability(device))
else:
    device = torch.device("cpu")
    User_params['device']['device_type'] = str(device)
print("Using device : ", device)
#=================================================================================
[train_dataloader, valid_dataloader, test_dataloader], label_dct = data.load_dataset(User_params, root_dir, root_dir_data, np_dtype)
#=================================================================================
if User_params['surrogate_backward_mode'] == "sigmoid" :
    surrogate_sigma = User_params['surrogate_sigma_sigmoid']
elif User_params['surrogate_backward_mode'] == "rectangle" :
    surrogate_sigma = User_params['surrogate_sigma_rec']
elif User_params['surrogate_backward_mode'] == "fast_sigmoid_abs" :
    surrogate_sigma = User_params['surrogate_a_fast_sigmoid_abs']
elif User_params['surrogate_backward_mode'] == "fast_sigmoid_tanh" :
    surrogate_sigma = User_params['surrogate_a_fast_sigmoid_tanh']
else :
    print(termcolor.colored('Error : surrogate_backward_mode value is wrong!', 'red'))

spike_fn_temp = models.SurrogateHeaviside
spike_fn_temp.backward_mode=User_params['surrogate_backward_mode']
spike_fn_temp.surrogate_mode=User_params['Train_params']['train_surrogate_mode']
spike_fn = spike_fn_temp.apply

layers = []

if User_params['Problem']=="All_with_encryption_feature" or User_params['Problem']=="Application_with_encryption_feature":
    input_shape = User_params['Dataset_params']['nb_bins_size'] + 3
else :
    input_shape = User_params['Dataset_params']['nb_bins_size']
output_shape = User_params['nb_dense_layer']
if User_params['High_speed_mode'] :
    w_init_mean = 0
    w_init_std = 1
else :
    w_init_mean=User_params['w_init_mean']
    w_init_std=User_params['w_init_std']
layers.append(models.SpikingDenseLayer(input_shape, output_shape, spike_fn,
                 w_init_mean=w_init_mean, w_init_std=w_init_std, spike_params=User_params['spike_params'], train_params=User_params['Train_params'],
                    surrogate_sigma=surrogate_sigma, recurrent=False, lateral_connections=User_params['lateral_connections'],
                    High_speed_mode=User_params['High_speed_mode'], Regularization_Term=User_params['Regularization_Term']))

input_shape = output_shape
output_shape = User_params['nb_outputs']
if User_params['High_speed_mode'] :
    w_init_mean = 0
    w_init_std = 0.1
else :
    w_init_mean=User_params['w_init_mean']
    w_init_std=User_params['w_init_std']
layers.append(models.ReadoutLayer(input_shape, output_shape, spike_fn,
                 w_init_mean=w_init_mean, w_init_std=w_init_std, spike_params=User_params['spike_params'], train_params=User_params['Train_params'],
                    surrogate_sigma=surrogate_sigma, time_reduction=User_params['Readout_time_reduction'], threshold=1))

snn = models.SNN(layers).to(device, dtype)

if (User_params['Train_mode'] or User_params['Test_mode']):
    utils.print_model_output(snn, User_params, train_dataloader, device, dtype)
# =================================================================================

def train(model, params, optimizer, train_dataloader, valid_dataloader, last_checkpoint=None, scheduler=None):

    log_softmax_fn = torch.nn.LogSoftmax(dim=1)
    loss_fn = torch.nn.NLLLoss()

    # if User_params['warmup_epochs'] > 0:
        # for g in optimizer.param_groups:
        # # g['lr'] /= len(train_dataloader)*User_params['warmup_epochs']
        # g['lr'] /= 40*User_params['warmup_epochs']
        # warmup_itr = 1

    hist = {'loss': [], 'valid_accuracy': [], 'epoch_train_time': [], 'validation_time':[]}
    hist['average_precision'] = []
    if last_checkpoint != None:
        checkpoint = last_checkpoint
        last_epochs = checkpoint['epoch']
        hist['loss'] = checkpoint['Epoch_loss_list']
        hist['valid_accuracy'] = checkpoint['Epoch_valid_accuracy_list']
        hist['epoch_train_time'] = checkpoint['epoch_train_time']
        hist['validation_time'] = checkpoint['validation_time']
        hist['average_precision'] = checkpoint['Epoch_average_precision_list']
        train_loss = checkpoint['Train_loss_list']
        train_reg_loss = checkpoint['Train_reg_loss_list']
    else :
        checkpoint = {}
        last_epochs = 0
        train_loss = []
        train_reg_loss = [[] for _ in range(len(model.layers) - 1)]

    for e in range(User_params['nb_epochs']):
        if e<last_epochs :
            print("Epoch %i has been done in last checkpoint" %(e + 1))
            continue

        if User_params['Readout_time_reduction'] == 'latency':
            if e < User_params['warmup_epochs']:
                model.layers[-1].time_reduction = User_params['warmup_Readout_mode']
            elif e == User_params['warmup_epochs']:
                model.layers[-1].latency_mode_init(device=device, dtype=dtype)
            else :
                model.layers[-1].time_reduction = "latency"

            model.layers[-1].print_params()
        if e >= User_params['warmup_epochs'] :
            if User_params['surrogate_sigma_scale']!=1:
                for i, l in enumerate(model.layers):
                    try:
                        state_dict = l.state_dict()
                        state_dict["surrogate_sigma"] = l.surrogate_sigma.detach() * User_params['surrogate_sigma_scale']
                        l.load_state_dict(state_dict)
                        print("Scaled surrogate_sigma in layer : ", i+1)
                    except:
                        continue

        start_time = time.time()
        local_loss = []
        reg_loss = [[] for _ in range(len(model.layers) - 1)]

        Batch_Num = 0
        for x_batch, y_batch in train_dataloader:
        # for x_batch, y_batch in utils.data_generator(x_train, y_train, User_params['batch_size'], device, dtype):
            Batch_Num += 1

            x_batch = x_batch.to(device, dtype)
            y_batch = y_batch.to(device)

            output, loss_seq = model(x_batch)
            if device == torch.device("cuda") and Batch_Num<2 and User_params['Readout_time_reduction'] == 'latency':
                print(output)
            log_p_y = log_softmax_fn(output)
            loss_val = loss_fn(log_p_y, y_batch.long())

            for i, loss in enumerate(loss_seq[:-1]):
                if type(loss)==list :
                    loss_list = loss
                    loss = loss_list[0]
                    if model.layers[-1].time_reduction == "latency":
                        spk_spread_loss = torch.zeros(size=(), dtype=dtype, device=device, requires_grad=False)
                    else :
                        spk_spread_loss = loss_list[1]
                else :
                    spk_spread_loss = torch.zeros(size=(), dtype=dtype, device=device, requires_grad=False)
                reg_loss_val = User_params['reg_loss_coef'] * loss * (i + 1) / len(loss_seq[:-1])
                # reg_loss_val += User_params['spk_spread_loss_coef']*spk_spread_loss
                reg_loss_val += User_params['spk_spread_loss_coef']*spk_spread_loss * (i + 1) / len(loss_seq[:-1])
                loss_val += reg_loss_val
                reg_loss[i].append(reg_loss_val.item())
                train_reg_loss[i].append(reg_loss_val.item())

            local_loss.append(loss_val.item())
            train_loss.append(loss_val.item())

            optimizer.zero_grad()
            loss_val.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step()
            model.clamp()

            # if e < User_params['warmup_epochs']:
            #     for g in optimizer.param_groups:
            #     g['lr'] *= (warmup_itr+1)/(warmup_itr)
            #     warmup_itr += 1

            checkpoint['epoch'] = e+1
            checkpoint['Batch_Num'] = Batch_Num
            # checkpoint['model_params'] = model.parameters()
            # checkpoint['model'] = model
            checkpoint['model_state_dict'] = model.state_dict()
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint['Train_loss_list'] = train_loss
            checkpoint['Train_reg_loss_list'] = train_reg_loss
            torch.save(checkpoint, Save_path_batch)

            if device == torch.device("cuda"):
                Batch_Num_print = (5*(int(User_params['nb_epochs']*User_params['number_of_batches']/(5*500))+1))
            elif device == torch.device("cpu"):
                Batch_Num_print = 1
            if(Batch_Num%Batch_Num_print==0):
                print("Batch %i: loss=%.5f, reg_loss=%.5f, %.1f percent of epoch completed" % (
                    Batch_Num, loss_val.item(), np.mean(np.array(reg_loss)[:,-1]), 100 * (Batch_Num / User_params['number_of_batches'])))

            if Batch_Num>=User_params['Batch_num_limit'] :
                break

        end_time = time.time()
        hist['epoch_train_time'].append(end_time-start_time)
        print("Epoch %i: Execution time=%.1f" % (e + 1, end_time-start_time))

        mean_loss = np.mean(local_loss)
        hist['loss'].append(mean_loss)
        print("Epoch %i: loss=%.5f" % (e + 1, mean_loss))

        if scheduler is not None and e >= User_params['warmup_epochs']:
            scheduler.step(mean_loss)

        for i, loss in enumerate(reg_loss):
            mean_reg_loss = np.mean(loss)
            print("Layer %i: reg loss=%.5f" % (i, mean_reg_loss))

        for i, l in enumerate(snn.layers[:-1]):
            try :
                print("Layer {}: average number of spikes={:.4f}".format(i, User_params['nb_steps'] * l.spk_rec_hist.mean()))
            except :
                print("Layer {}: No spikes exist in this layer!".format(i))

        if User_params['Validation_mode'] :
            start_time = time.time()
            # valid_dataloader_temp = utils.sparse_data_generator(x_test, y_test, User_params['batch_size'], User_params['nb_steps'], User_params['nb_inputs'], User_params['time_step'],device, dtype)
            # valid_dataloader_temp = utils.data_generator(x_test, y_test, User_params['batch_size'], device, dtype)
            valid_dataloader_temp = valid_dataloader
            confusion_matrix = compute_confusion_matrix(model, valid_dataloader_temp)
            valid_accuracy, average_precision = compute_and_print_score_categories(confusion_matrix)
            end_time = time.time()
            hist['validation_time'].append(end_time-start_time)
            hist['valid_accuracy'].append(valid_accuracy)
            print("Validation accuracy=%.3f" % (valid_accuracy))
            hist['average_precision'].append(average_precision)
            print("Average precision = %.1f"%(average_precision)+"%")

        checkpoint['Epoch_loss_list'] = hist['loss']
        checkpoint['Epoch_valid_accuracy_list'] = hist['valid_accuracy']
        checkpoint['Epoch_average_precision_list'] = hist['average_precision']
        checkpoint['epoch_train_time'] = hist['epoch_train_time']
        checkpoint['validation_time'] = hist['validation_time']
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint, Save_path_epoch)

        if (e+1)%User_params['snapshot_epochs'][0]==0 or (e+1) in User_params['snapshot_epochs']:
            model_id = uuid.uuid4()
            checkpoint['model_id'] = model_id
            torch.save(checkpoint, Save_path_final+str(model_id)+".pth")
            if User_params['Results_save'] :
                utils.save_results(model, checkpoint, User_params, Results_path)

    return hist, checkpoint


def compute_classification_accuracy(model, dataloader):
    accs = []
    spike_fn_temp = models.SurrogateHeaviside
    spike_fn_temp.backward_mode = User_params['surrogate_backward_mode']
    spike_fn_temp.surrogate_mode = User_params['Train_params']['test_surrogate_mode']
    spike_fn = spike_fn_temp.apply
    for l in model.layers:
        try:
            l.spike_fn = spike_fn
            l.training = False
        except:
            continue

    with torch.no_grad():
        for x_batch, y_batch in dataloader:

            # x_batch = torch.tensor(x_batch, dtype=dtype, device=device, requires_grad=False)
            # x_batch = x_batch.to_dense()
            x_batch = x_batch.to(device, dtype)
            x_batch = torch.reshape(x_batch, (User_params['batch_size'], User_params['Input_channels'], User_params['image_H'], User_params['image_W']))
            # x_batch = torch.reshape(x_batch, (User_params['batch_size'], 1, 1, User_params['image_H'], User_params['image_W']))
            # x_batch = x_batch.repeat(1, 1, User_params['nb_steps'], 1, 1)
            # x_batch = torch.reshape(x_batch, (User_params['batch_size'], 1, User_params['nb_steps'], User_params['nb_inputs']))

            y_batch = y_batch.to(device)
            output, _ = model(x_batch)
            _,am=torch.max(output,1) # argmax over output units
            tmp = np.mean((y_batch.long()==am).detach().cpu().numpy()) # compare to labels
            accs.append(tmp)

    spike_fn_temp = models.SurrogateHeaviside
    spike_fn_temp.backward_mode = User_params['surrogate_backward_mode']
    spike_fn_temp.surrogate_mode = User_params['Train_params']['train_surrogate_mode']
    spike_fn = spike_fn_temp.apply
    for l in model.layers:
        try:
            l.spike_fn = spike_fn
            l.training = True
        except:
            continue

    return np.mean(accs)
    

def compute_confusion_matrix(model, dataloader, percent=False):
    confusion_matrix = np.zeros((User_params['nb_outputs'], User_params['nb_outputs'])).astype(int)
    
    with torch.no_grad():

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device, dtype)

            # # idx = torch.randperm(User_params['nb_steps'])
            # # x_batch = x_batch[:, idx, :]
            # # print(x_batch.shape)
            # for i in range(300) :
                # # idx = torch.randperm(User_params['nb_steps'])
                # # x_batch[:, :, i] = x_batch[:, idx, i]
                # for j in range(int(1500-2)):
                # for j in range(int(1500/3)):
                    # idx = torch.randperm(3)
                    # x_batch[:, j:j+3, i] = x_batch[:, j+idx, i]
                    # x_batch[:, (3*j):(3*(j+1)), i] = x_batch[:, (3*j)+idx, i]

            y_batch = y_batch.to(device)
            output, _ = model(x_batch)
            _, am=torch.max(output,1) # argmax over output units

            for i in range(len(y_batch)):
                confusion_matrix[am[i],y_batch[i]] += 1
    if percent:
        confusion_matrix = confusion_matrix.astype(float) / confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix = np.round(confusion_matrix * 100, 2)

    return confusion_matrix


def compute_and_print_score_categories(confusion_matrix):

    if User_params['Problem'] == "All" or User_params['Problem']=="All_with_encryption_feature":
        categ_name = {0:'None', 1:'Botnet', 2:'Exploit', 3:'Infiltration', 5:'Benign', 6:'Malicious'}
        categ_list = {0:[0], 1:[1,2,3]}

    Avg_Pr = []
    Total_Acc = []
    tot_data = np.sum(confusion_matrix)

    print(15*"-"+"\nConfusion matrix:")
    print(confusion_matrix)


    print(15 * "-")
    print("One vs All results for different traffic classes and encryption techniques (Flowpic- Table3-rows4-6):")
    print(15 * "-")
    for i in range(len(categ_list)):
        TP,FP,FN = 0,0,0
        for ind_1 in categ_list[i]:
          FP += np.sum(confusion_matrix[ind_1,:])
          FN += np.sum(confusion_matrix[:,ind_1])
          for ind_2 in categ_list[i]:
            TP += confusion_matrix[ind_1,ind_2]
        FP -= TP
        FN -= TP
        TN = tot_data-TP-FP-FN
        Ac, Pr, Re = compute_metrics(TP,TN,FP,FN)
        print(categ_name[i] + " : Re" + " = %.1f"%(Re*100)+"%" + " ; Pr" + " = %.1f"%(Pr*100)+"%" " ; Ac" + " = %.1f"%(Ac*100)+"%")


    print(15 * "-")
    print("One vs All results for all categories :")
    print(15 * "-")
    for ind in range(User_params['nb_outputs']):
        tot_categ = np.sum(confusion_matrix[:,ind])
        TP = confusion_matrix[ind, ind]
        FP = np.sum(confusion_matrix[ind,:])-TP
        FN = tot_categ-TP
        TN = tot_data-TP-FP-FN
        Ac, Pr, Re = compute_metrics(TP,TN,FP,FN)
        Total_Acc.append(TP/tot_data)
        Avg_Pr.append(Pr*tot_categ/tot_data)
        print(str(utils.key(label_dct,ind)) + " (" + str(ind) + ") : Re" + " = %.1f"%(Re*100)+"%" + " ; Pr" + " = %.1f"%(Pr*100)+"%" " ; Ac" + " = %.1f"%(Ac*100)+"%")


    print(15 * "-")
    print("Results for classification of traffic categories when input data is just Non-VPN/VPN/Tor (Flowpic- Table3-rows1-3, Table4):")
    print(15 * "-")
    for i in range(len(categ_list)):
        if (categ_name[i] == "nonVPN" or categ_name[i] == "VPN" or categ_name[i] == "Tor"):
            confusion_matrix_temp = np.zeros((len(categ_list[i]), len(categ_list[i]))).astype(int)
            
            for ind_1 in range(len(categ_list[i])):
                for ind_2 in range(len(categ_list[i])):
                    for ind_3 in categ_list[ind_1] :
                        confusion_matrix_temp[ind_1,ind_2] += confusion_matrix[ind_3, categ_list[i][ind_2]]
            # print(confusion_matrix_temp.astype(int))
            Total_Acc_temp, Avg_Pr_temp, Ac, Pr, Re = compute_matrix_total_metrics(confusion_matrix_temp)
            
            print("Multi-class for " + categ_name[i] + " : Avg_Pr" + " = %.1f"%(Avg_Pr_temp*100)+"%" " ; Total_Acc" + " = %.1f"%(Total_Acc_temp*100)+"%")
            print("One vs all for Video-" + categ_name[i] + " : Re" + " = %.1f"%(Re[0]*100)+"%" + " ; Pr" + " = %.1f"%(Pr[0]*100)+"%" " ; Ac" + " = %.1f"%(Ac[0]*100)+"%")
            print("One vs all for VoIP-" + categ_name[i] + " : Re" + " = %.1f"%(Re[1]*100)+"%" + " ; Pr" + " = %.1f"%(Pr[1]*100)+"%" " ; Ac" + " = %.1f"%(Ac[1]*100)+"%")
            print("One vs all for F.T.-" + categ_name[i] + " : Re" + " = %.1f"%(Re[2]*100)+"%" + " ; Pr" + " = %.1f"%(Pr[2]*100)+"%" " ; Ac" + " = %.1f"%(Ac[2]*100)+"%")
            print("One vs all for Chat-" + categ_name[i] + " : Re" + " = %.1f"%(Re[3]*100)+"%" + " ; Pr" + " = %.1f"%(Pr[3]*100)+"%" " ; Ac" + " = %.1f"%(Ac[3]*100)+"%")
            if (not categ_name[i] == "VPN"):
                print("One vs all for Browsing-" + categ_name[i] + " : Re" + " = %.1f"%(Re[4]*100)+"%" + " ; Pr" + " = %.1f"%(Pr[4]*100)+"%" " ; Ac" + " = %.1f"%(Ac[4]*100)+"%")

            # confusion_matrix_temp = np.copy(confusion_matrix)
            # for ind_1 in range(User_params['nb_outputs']):
                # for ind_2 in range(User_params['nb_outputs']):
                    # if not (ind_1 in categ_list[i]) or not (ind_2 in categ_list[i]):
                        # confusion_matrix_temp[ind_1, ind_2] = 0

    """adequar trecho abaixo"""
    # print(15 * "-")
    # print("Results for multi-class classification of Encryption techniques (Flowpic- Table3-row7):")
    # print(15 * "-")

    # confusion_matrix_temp = np.zeros((3,3)).astype(int)
    # nonVPN_list = categ_list[5]
    # Tor_list = categ_list[6]
    # VPN_list = categ_list[7] #7

    # for ind_1 in range(User_params['nb_outputs']):
    #     for ind_2 in range(User_params['nb_outputs']):
    #         if ind_1 in nonVPN_list and ind_2 in nonVPN_list :
    #             confusion_matrix_temp[0,0] += confusion_matrix[ind_1,ind_2]
    #         elif ind_1 in nonVPN_list and ind_2 in VPN_list :
    #             confusion_matrix_temp[0,1] += confusion_matrix[ind_1,ind_2]
    #         elif ind_1 in nonVPN_list and ind_2 in Tor_list :
    #             confusion_matrix_temp[0,2] += confusion_matrix[ind_1,ind_2]
            
    #         elif ind_1 in VPN_list and ind_2 in nonVPN_list :
    #             confusion_matrix_temp[1,0] += confusion_matrix[ind_1,ind_2]
    #         elif ind_1 in VPN_list and ind_2 in VPN_list :
    #             confusion_matrix_temp[1,1] += confusion_matrix[ind_1,ind_2]
    #         elif ind_1 in VPN_list and ind_2 in Tor_list :
    #             confusion_matrix_temp[1,2] += confusion_matrix[ind_1,ind_2]

    #         elif ind_1 in Tor_list and ind_2 in nonVPN_list :
    #             confusion_matrix_temp[2,0] += confusion_matrix[ind_1,ind_2]
    #         elif ind_1 in Tor_list and ind_2 in VPN_list :
    #             confusion_matrix_temp[2,1] += confusion_matrix[ind_1,ind_2]
    #         elif ind_1 in Tor_list and ind_2 in Tor_list :
    #             confusion_matrix_temp[2,2] += confusion_matrix[ind_1,ind_2]

    # Total_Acc_temp, Avg_Pr_temp, Ac, Pr, Re = compute_matrix_total_metrics(confusion_matrix_temp)
    
    # print("multi-class classification of Encryption techniques : " + " Avg_Pr" + " = %.1f"%(Avg_Pr_temp*100)+"%" " ; Total_Acc" + " = %.1f"%(Total_Acc_temp*100)+"%")
    # print("One vs all for nonVPN" + " : Re" + " = %.1f"%(Re[0]*100)+"%" + " ; Pr" + " = %.1f"%(Pr[0]*100)+"%" " ; Ac" + " = %.1f"%(Ac[0]*100)+"%")
    # print("One vs all for VPN" + " : Re" + " = %.1f"%(Re[1]*100)+"%" + " ; Pr" + " = %.1f"%(Pr[1]*100)+"%" " ; Ac" + " = %.1f"%(Ac[1]*100)+"%")
    # print("One vs all for Tor" + " : Re" + " = %.1f"%(Re[2]*100)+"%" + " ; Pr" + " = %.1f"%(Pr[2]*100)+"%" " ; Ac" + " = %.1f"%(Ac[2]*100)+"%")



    # print(15 * "-")
    # print("Final results :")
    # print(15 * "-")

    return(np.sum(Total_Acc)*100, np.sum(Avg_Pr)*100)


def compute_matrix_total_metrics(confusion_matrix):
    tot_data = np.sum(confusion_matrix)
    Avg_Pr = []
    Total_Acc = []
    matrix_shape = np.shape(confusion_matrix)

    Ac = np.zeros((matrix_shape[0]))
    Pr = np.zeros((matrix_shape[0]))
    Re = np.zeros((matrix_shape[0]))
    for ind in range(matrix_shape[0]):
        tot_categ = np.sum(confusion_matrix[:,ind])
        TP = confusion_matrix[ind, ind]
        FP = np.sum(confusion_matrix[ind,:])-TP
        FN = tot_categ-TP
        TN = tot_data-TP-FP-FN
        Ac[ind], Pr[ind], Re[ind] = compute_metrics(TP,TN,FP,FN)
        Total_Acc.append(TP/tot_data)
        Avg_Pr.append(Pr*tot_categ/tot_data)
    return np.sum(Total_Acc), np.sum(Avg_Pr), Ac, Pr, Re


def compute_metrics(TP,TN,FP,FN):
    if TP==0 and FP==0:
        Ac, Pr, Re = (TP+TN)/(TP+FP+FN+TN), 0, TP/(TP+FN)
    else:
        Ac, Pr, Re = (TP+TN)/(TP+FP+FN+TN), TP/(TP+FP), TP/(TP+FN)
    return Ac, Pr, Re
# =================================================================================

checkpoint = None
if User_params['Model_load'] :
    if (os.path.exists(Load_path)):
        checkpoint = torch.load(Load_path, map_location=device, weights_only=False)
        snn.load_state_dict(checkpoint['model_state_dict'])
        # params = checkpoint['model_params']
        print("Saved model loaded")
    else:
        print("No model exists to load!!!")


if User_params['Train_mode'] :
    params = []
    if User_params['Train_params']['w']:
        params = [{'params':l.w, 'lr':User_params['lr'], "weight_decay":User_params['weight_decay'] } for i,l in enumerate(snn.layers) if models.layerHasParamsW(l)]
        params += [{'params':l.w_dw, 'lr':User_params['lr'], "weight_decay":User_params['weight_decay']} for i,l in enumerate(snn.layers) if models.isSeperableConvlayer(l)]
        params += [{'params':l.w_pw, 'lr':User_params['lr'], "weight_decay":User_params['weight_decay']} for i,l in enumerate(snn.layers) if models.isSeperableConvlayer(l)]
    if User_params['Train_params']['v']:
        params += [{'params':l.v, 'lr':User_params['lr'], "weight_decay":User_params['weight_decay']} for i,l in enumerate(snn.layers[:-1]) if models.layerHasParamsV(l) and l.recurrent]
    if User_params['Train_params']['b']:
        params += [{'params':l.b, 'lr':User_params['lr']} for i,l in enumerate(snn.layers) if models.layerHasParamsB(l)]
    if User_params['Train_params']['beta']:
        params += [{'params': l.beta, 'lr': User_params['lr']} for i, l in enumerate(snn.layers) if models.layerHasParamsBeta(l)]
    if User_params['Train_params']['surrogate_sigma']:
        params += [{'params': l.surrogate_sigma, 'lr': User_params['lr']} for i, l in enumerate(snn.layers) if models.layerHasHeaviside(l)]
    if User_params['Train_params']['BN_scale']:
        params += [{'params': l.scale, 'lr': User_params['lr']} for i, l in enumerate(snn.layers) if models.isBatchNormLayer(l)]
    if User_params['Train_params']['BN_offset']:
        params += [{'params': l.offset, 'lr': User_params['lr']} for i, l in enumerate(snn.layers) if models.isBatchNormLayer(l)]
    if User_params['Train_params']['Readout_b_latency']:
        params += [{'params': snn.layers[-1].b_latency, 'lr': User_params['lr']}]
    if User_params['Train_params']['Readout_latency_scale']:
        params += [{'params': snn.layers[-1].latency_scale, 'lr': User_params['lr']}]

    if User_params['optimizer']=="RAdam":
        optimizer = optim.RAdam(params)
    elif User_params['optimizer']=="Adam":
        optimizer = torch.optim.Adam(params, lr=User_params['lr'], betas=User_params['betas'], eps=User_params['eps'], weight_decay=User_params['weight_decay'])
    elif User_params['optimizer'] == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=User_params['lr'], betas=User_params['betas'], eps=User_params['eps'], weight_decay=User_params['weight_decay'])
    elif User_params['optimizer'] == "Adamax":
        optimizer = torch.optim.Adamax(params, lr=User_params['lr'], betas=User_params['betas'])
    else :
        print("Error : No optimizer specified!")

    Scheduler_params = User_params['Scheduler_params']
    if User_params['scheduler']=="ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, User_params['gamma'], last_epoch=-1, verbose=True)
    elif User_params['scheduler'] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=Scheduler_params['mode'], factor=Scheduler_params['factor'], patience=Scheduler_params['patience'], verbose=True,
                    threshold=Scheduler_params['threshold'], threshold_mode=Scheduler_params['threshold_mode'], cooldown=Scheduler_params['cooldown'], min_lr=Scheduler_params['min_lr'],
                    eps=Scheduler_params['eps'])
    elif User_params['scheduler'] == "CosineAnnealingLR" :
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Scheduler_params['T_max'], eta_min=Scheduler_params['min_lr'], last_epoch=-1, verbose=True)
    else :
        scheduler = None

    if (User_params['Model_load'] == True):
        if (os.path.exists(Load_path)):
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except : None

    if User_params['Train_Profiler'] :
        with profile(use_cuda=use_cuda) as prof :
            hist, checkpoint = train(snn, params, optimizer, train_dataloader, valid_dataloader,
                         last_checkpoint=checkpoint, scheduler=scheduler)
        print("Output of key_averages :")
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        print("Profile :")
        print(prof)
    else :
        hist, checkpoint = train(snn, params, optimizer, train_dataloader, valid_dataloader,
                     last_checkpoint=checkpoint, scheduler=scheduler)
    if device == torch.device("cuda"):
        print("torch memory_stats : ",torch.cuda.memory_stats(device))
        print("torch memory_allocated : ",torch.cuda.memory_allocated(device))
        print("torch max_memory_allocated : ",torch.cuda.max_memory_allocated(device))
    else :
        pid = os.getpid()
        py = psutil.Process(pid)
        # print("CPU percent : ", py.cpu_percent())
        print("CPU memory percent : ", py.memory_percent())
        print("CPU Memory use(GB) : ", py.memory_info()[0] / 2. ** 30) # memory use in GB...I think
# =================================================================================

if User_params['Test_mode'] :

    if (User_params['Readout_time_reduction'] == 'latency') :
        snn.layers[-1].latency_mode_init(device=device, dtype=dtype)
        snn.layers[-1].print_params()

    # train_accuracy = compute_classification_accuracy(snn, train_dataloader)
    # print("Train accuracy=%.3f"%(train_accuracy))

    for i, l in enumerate(snn.layers):
        print("Beta for Layer {}: {}".format(i, l.beta))

    if User_params['Test_Profiler'] :
        with profile(use_cuda=use_cuda) as prof:
            test_confusion_matrix = compute_confusion_matrix(snn, test_dataloader, percent=True)
            cm = compute_confusion_matrix(snn, test_dataloader)
            test_accuracy, average_precision = compute_and_print_score_categories(cm)

            print("Test accuracy=%.3f"%(test_accuracy))
            print("Average precision = %.1f"%(average_precision)+"%")
        print("Output of key_averages :")
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        print("Profile :")
        print(prof)
    else :
        test_confusion_matrix = compute_confusion_matrix(snn, test_dataloader, percent=True)
        cm = compute_confusion_matrix(snn, test_dataloader)
        test_accuracy, average_precision = compute_and_print_score_categories(cm)

        print("Test accuracy=%.3f"%(test_accuracy))
        print("Average precision = %.1f"%(average_precision)+"%")

    if checkpoint!=None and User_params['Results_save'] and utils.json_exist(checkpoint['model_id'], Results_path):
        # checkpoint['train_accuracy'] = train_accuracy
        checkpoint['test_accuracy'] = test_accuracy
        utils.save_results(snn, checkpoint, User_params, Results_path)
# =================================================================================

if (User_params['Plot_graphs']):
    # utils.plot_model_output(snn, User_params, test_dataloader, label_dct, dtype)
    utils.plot_model_behaviour_grid_columns(snn, User_params, test_dataloader, label_dct, device, dtype)

    plt.figure(figsize=(6, 5))
    sns.heatmap(test_confusion_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=label_dct.keys(), yticklabels=label_dct.keys())

    # Adding labels
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.title('Confusion Matrix Heatmap')

# =================================================================================
if User_params['Plot_graphs'] :
    if checkpoint != None:
        utils.plot_loss_acc_epoch_charts(User_params, checkpoint)

# =================================================================================

if User_params['Find_model'] :
    utils.find_model(User_params, Find_path)

# =================================================================================

if (User_params['plot_show'] == True):
    plt.show()