from dataset import *
import glob
from pandas.core.common import flatten
import torch, torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import Normalize
import cv2

def get_shadow_object_paths(image_list):
    shadow_list = []
    object_list = []
    for path in image_list:
        shadow_path_split = path.split("/")
        shadow_path_split[-4] += "_shadow"
        shadow_list.append(os.path.join(*shadow_path_split))

        object_path_split = path.split("/")
        object_path_split[-4] += "_object"
        object_list.append(os.path.join(*object_path_split))
    return shadow_list, object_list

def load_all_paths(base_path, data_paths):
    all_test_paths = []
    
    for data_path in data_paths:
        data_path = base_path + data_path

        test_paths = []
        for path in glob.glob(data_path):
            test_paths.append(glob.glob(path + "/test/*/*"))
        test_paths = list(flatten(test_paths))
        all_test_paths += test_paths
    
    all_test_paths.sort()
    return all_test_paths


def apply_gradcam(model, image, target_layer):
    # Convert 2-channel grayscale to 3-channel RGB
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image.unsqueeze(0))[0]
    visualization = show_cam_on_image(image.detach().permute(1, 2, 0).cpu().numpy(), grayscale_cam, use_rgb=True)
    return visualization

def test(model, test_dataloader, save_path, test_type):

    correct = 0
    total = 0
    model.eval()
    all_predicted = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    all_pred_probs = torch.tensor([]).to(device)
    target_layer = model.layer4[-1]
    i = 0
    for images, labels in tqdm(test_dataloader, desc="testing"):
        if i >= 5:  
            break
        i += 1
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted = torch.cat((all_predicted, predicted))
            all_labels = torch.cat((all_labels, labels))
            all_pred_probs = torch.cat((all_pred_probs, outputs.data))

        # Denormalize the image
        # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        mean = torch.tensor([0.485, 0.456]).view(2, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224]).view(2, 1, 1).to(device)
        
        print(images[0].shape)
        print(std.shape)
        print(mean.shape)
        denorm_image = images[0] * std + mean
        denorm_image = denorm_image.requires_grad_()
        
        gradcam_vis = apply_gradcam(model, denorm_image, target_layer)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        
        plt.imshow(denorm_image.detach()[0].cpu().numpy(), cmap='gray')
        plt.title(f"Original - True: {idx_to_class[labels[0].item()]}, Pred: {idx_to_class[predicted[0].item()]}")
        plt.subplot(1, 2, 2)
        plt.imshow(gradcam_vis)
        plt.title("GradCAM Visualization")
        plot_name = save_path.split("/")[-1].split(".")[-2]
        plt.savefig(f"./plots/gradcam_{plot_name}_{test_type}_{i}.png")
        plt.close()

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(all_labels.cpu(), all_pred_probs[:,1].cpu())
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plot_name = save_path.split("/")[-1].split(".")[-2]
    plt.title(f'ROC: {plot_name}_{test_type}')
    plt.legend(loc="lower right")
    plt.savefig(f"./plots/{plot_name}_{test_type}.png")
    with open(f"./plots/{plot_name}_{test_type}.pkl", 'wb') as f:
        pickle.dump([fpr, tpr, roc_auc], f)

    conf_matrix = confusion_matrix(all_labels.cpu(), all_predicted.cpu())
    print(f"{plot_name} {test_type}")
    print("ROC curve area:", roc_auc)
    print(conf_matrix)
    print(f"{conf_matrix[0].sum().item()} generated images, {conf_matrix[1].sum().item()} real images")
    tn = conf_matrix[0,0]
    tp = conf_matrix[1,1]
    fp = conf_matrix[0,1]
    fn = conf_matrix[1,0]
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Precision: {tp/(tp+fp)}, Recall: {tp/(tp+fn)}")
    accuracy = 100 * correct / total
    print(f"Accuracy for {save_path}: {accuracy}")
    print()

def test_ShadowObjectCNN():
    print("ShadowObjectResnet:")
    if category == "indoor":
        save_path = "./checkpoints/ShadowObject_indoor.pth"
    elif category == "outdoor":
        save_path = "./checkpoints/ShadowObject_outdoor.pth"
    elif category == "combined":
        save_path = "./checkpoints/ShadowObject_combined.pth"

    model = torchvision.models.resnet50(weights = None)
    model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    try:
        if device == "cpu":
            print("Innn")
            model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu'), weights_only=True))
        else:
            model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu'), weights_only=True))
        print("Successfully Loaded Saved Model")
    except Exception as error:
        print("Failed to load Saved Model")
        print(error)
    model.to(device)

    easy_test_shadow = sorted(list(set(all_test_shadow)-set(unconfident_shadow)-set(misclassified_shadow)))
    easy_test_object = sorted(list(set(all_test_object)-set(unconfident_object)-set(misclassified_object)))
    test_set = ShadowObjectDataset(easy_test_shadow, easy_test_object, class_to_idx)
    test_dataloader = DataLoader(test_set, batch_size = 128, shuffle = False, num_workers=6)
    # test(model, test_dataloader, save_path, "easy")

    test_set = ShadowObjectDataset(unconfident_shadow, unconfident_object, class_to_idx)
    test_dataloader = DataLoader(test_set, batch_size = 128, shuffle = False, num_workers=6)
    test(model, test_dataloader, save_path, "unconfident")

    test_set = ShadowObjectDataset(misclassified_shadow, misclassified_object, class_to_idx)
    test_dataloader = DataLoader(test_set, batch_size = 128, shuffle = False, num_workers=6)
    test(model, test_dataloader, save_path, "misclassified")

if __name__ == "__main__":

    idx_to_class = {0: 'real', 1: 'gen'}
    class_to_idx = {value:key for key,value in idx_to_class.items()}
    print(class_to_idx)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--category", type=str, default="indoor", help='choose: indoor, outdoor or combined')
    args = parser.parse_args()

    category = args.category
    print("category:", category)

    base_path = "../dataset/"

    misclassified_indoor_file = "./misclassified_indoor_list.pkl"
    misclassified_outdoor_file = "./misclassified_outdoor_list.pkl"

    unconfident_indoor_file = "./unconfident_indoor_list.pkl"
    unconfident_outdoor_file = "./unconfident_outdoor_list.pkl"

    if category == "indoor":
        image_data_paths = ["Kandinsky_Indoor"]
        shadow_data_paths = ["Kandinsky_Indoor_shadow"]
        object_data_paths = ["Kandinsky_Indoor_object"]

        misclassified_image_list = sorted(pickle.load(open(misclassified_indoor_file, "rb")))
        unconfident_image_list = sorted(pickle.load(open(unconfident_indoor_file, "rb")))

    elif category == "outdoor":

        image_data_paths = ["Kandinsky_Outdoor"]
        shadow_data_paths = ["Kandinsky_Outdoor_shadow"]
        object_data_paths = ["Kandinsky_Outdoor_object"]

        misclassified_image_list = sorted(pickle.load(open(misclassified_outdoor_file, "rb")))
        unconfident_image_list = sorted(pickle.load(open(unconfident_outdoor_file, "rb")))

    elif category == "combined":
        image_data_paths = ["Kandinsky_Indoor", "Kandinsky_Outdoor"]
        shadow_data_paths = ["Kandinsky_Indoor_shadow", "Kandinsky_Outdoor_shadow"]
        object_data_paths = ["Kandinsky_Indoor_object", "Kandinsky_Outdoor_object"]

        misclassified_image_list = sorted(pickle.load(open(misclassified_indoor_file, "rb"))) + \
            sorted(pickle.load(open(misclassified_outdoor_file, "rb")))
        unconfident_image_list = sorted(pickle.load(open(unconfident_indoor_file, "rb"))) + \
            sorted(pickle.load(open(unconfident_outdoor_file, "rb")))

    all_test_image = load_all_paths(base_path, image_data_paths)
    all_test_shadow = load_all_paths(base_path, shadow_data_paths)
    all_test_object = load_all_paths(base_path, object_data_paths)

    misclassified_shadow, misclassified_object = get_shadow_object_paths(misclassified_image_list)
    unconfident_shadow, unconfident_object = get_shadow_object_paths(unconfident_image_list)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    test_ShadowObjectCNN()