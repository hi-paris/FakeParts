import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from tqdm import tqdm



def get_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    ap = average_precision_score(y_true, y_pred)

    return ap, acc, r_acc, f_acc


def validate(model, loader, dataset_name, result_folder, save_tensor=False, find_thres=False):

    with torch.no_grad():
        #y_true, y_pred = [], []
        print ("Number of batch: %d" %(len(loader)))
        print("Number of images: ", len(loader.dataset))
        print("Batch size: ", loader.batch_size)
        for i, (video_tensor, label) in enumerate(tqdm(loader, total=len(loader))):
            #print(f"Processing batch {i}")
            print("video_tensor shape: ", video_tensor.shape)

            #take the first element of the batch
            video = video_tensor[0]
            nb_frames = video.shape[0]

            video_path = loader.dataset.total_list[i]
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            in_tens = video.cuda()

            y_pred = model(in_tens).sigmoid().flatten().tolist()
            y_true = [label[0]]*nb_frames

            y_true, y_pred = np.array(y_true), np.array(y_pred)
            #print(f"y_true: {y_true.shape}, y_pred: {y_pred.shape}")

            #saving tensor
            if save_tensor:
                dataset_saving_folder = os.path.join(result_folder, dataset_name)
                real_video_saving_dir = os.path.join(dataset_saving_folder, '0_real')
                fake_video_saving_dir = os.path.join(dataset_saving_folder, '1_fake')

                #Creating directories
                if not os.path.exists(dataset_saving_folder):
                    os.makedirs(dataset_saving_folder)
                if not os.path.exists(real_video_saving_dir):
                    os.makedirs(real_video_saving_dir)
                if not os.path.exists(fake_video_saving_dir):
                    os.makedirs(fake_video_saving_dir)

                if label[0] == 0:
                    tensor_path = os.path.join(real_video_saving_dir, video_name+'_results.pth')
                else:
                    tensor_path = os.path.join(fake_video_saving_dir, video_name+'_results.pth')

                torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ), tensor_path )
                print(f"Tensor has been saved at {tensor_path}")

     
    '''
    # Get AP 
    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0


    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres
    '''

    return 1,1,1,1,1,1,1,1


############################################################################################################