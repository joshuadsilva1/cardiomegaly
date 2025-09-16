import numpy as np
import gradio as gr
import cv2 

from models.HybridGNet2IGSC import Hybrid 
from utils.utils import scipy_to_torch_sparse, genMatrixesLungsHeart
import scipy.sparse as sp
import torch
import pandas as pd
from zipfile import ZipFile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hybrid = None

def getDenseMask(landmarks, h, w):
    
    RL = landmarks[0:44]
    LL = landmarks[44:94]
    H = landmarks[94:]
    
    img = np.zeros([h, w], dtype = 'uint8')
    
    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')
    H = H.reshape(-1, 1, 2).astype('int')

    img = cv2.drawContours(img, [RL], -1, 1, -1)
    img = cv2.drawContours(img, [LL], -1, 1, -1)
    img = cv2.drawContours(img, [H], -1, 2, -1)
    
    return img

def getMasks(landmarks, h, w):
    
    RL = landmarks[0:44]
    LL = landmarks[44:94]
    H = landmarks[94:]
    
    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')
    H = H.reshape(-1, 1, 2).astype('int')
    
    RL_mask = np.zeros([h, w], dtype = 'uint8')
    LL_mask = np.zeros([h, w], dtype = 'uint8')
    H_mask = np.zeros([h, w], dtype = 'uint8')
    
    RL_mask = cv2.drawContours(RL_mask, [RL], -1, 255, -1)
    LL_mask = cv2.drawContours(LL_mask, [LL], -1, 255, -1)
    H_mask = cv2.drawContours(H_mask, [H], -1, 255, -1)

    return RL_mask, LL_mask, H_mask

def drawOnTop(img, landmarks, original_shape):
    h, w = original_shape
    output = getDenseMask(landmarks, h, w)
    
    image = np.zeros([h, w, 3])
    image[:,:,0] = img + 0.3 * (output == 1).astype('float') - 0.1 * (output == 2).astype('float')
    image[:,:,1] = img + 0.3 * (output == 2).astype('float') - 0.1 * (output == 1).astype('float') 
    image[:,:,2] = img - 0.1 * (output == 1).astype('float') - 0.2 * (output == 2).astype('float') 

    image = np.clip(image, 0, 1)
    
    RL, LL, H = landmarks[0:44], landmarks[44:94], landmarks[94:]
    
    # Draw the landmarks as dots
    
    for l in RL:
        image = cv2.circle(image, (int(l[0]), int(l[1])), 5, (1, 0, 1), -1)
    for l in LL:
        image = cv2.circle(image, (int(l[0]), int(l[1])), 5, (1, 0, 1), -1)
    for l in H:
        image = cv2.circle(image, (int(l[0]), int(l[1])), 5, (1, 1, 0), -1)
    
    return image
    

def loadModel(device):    
    A, AD, D, U = genMatrixesLungsHeart()
    N1 = A.shape[0]
    N2 = AD.shape[0]

    A = sp.csc_matrix(A).tocoo()
    AD = sp.csc_matrix(AD).tocoo()
    D = sp.csc_matrix(D).tocoo()
    U = sp.csc_matrix(U).tocoo()

    D_ = [D.copy()]
    U_ = [U.copy()]

    config = {}

    config['n_nodes'] = [N1, N1, N1, N2, N2, N2]
    A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]
    
    A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to(device) for x in X] for X in (A_, D_, U_))

    config['latents'] = 64
    config['inputsize'] = 1024

    f = 32
    config['filters'] = [2, f, f, f, f//2, f//2, f//2]
    config['skip_features'] = f

    hybrid = Hybrid(config.copy(), D_t, U_t, A_t).to(device)
    hybrid.load_state_dict(torch.load("weights/weights.pt", map_location=torch.device(device)))
    hybrid.eval()
    
    return hybrid


def pad_to_square(img):
    h, w = img.shape[:2]
    
    if h > w:
        padw = (h - w) 
        auxw = padw % 2
        img = np.pad(img, ((0, 0), (padw//2, padw//2 + auxw)), 'constant')
        
        padh = 0
        auxh = 0
        
    else:
        padh = (w - h) 
        auxh = padh % 2
        img = np.pad(img, ((padh//2, padh//2 + auxh), (0, 0)), 'constant')

        padw = 0
        auxw = 0
        
    return img, (padh, padw, auxh, auxw)
    

def preprocess(input_img):
    img, padding = pad_to_square(input_img)
    
    h, w = img.shape[:2]
    if h != 1024 or w != 1024:
        img = cv2.resize(img, (1024, 1024), interpolation = cv2.INTER_CUBIC)
        
    return img, (h, w, padding)


def removePreprocess(output, info):
    h, w, padding = info
    
    if h != 1024 or w != 1024:
        output = output * h
    else:
        output = output * 1024
    
    padh, padw, auxh, auxw = padding
    
    output[:, 0] = output[:, 0] - padw//2
    output[:, 1] = output[:, 1] - padh//2
    
    return output   


def zip_files(files):
    with ZipFile("complete_results.zip", "w") as zipObj:
        for idx, file in enumerate(files):
            zipObj.write(file, arcname=file.split("/")[-1])
    return "complete_results.zip"


def segment(input_img):
    global hybrid, device
    
    if hybrid is None:
        hybrid = loadModel(device)
    
    input_img = cv2.imread(input_img, 0) / 255.0
    original_shape = input_img.shape[:2]
    
    img, (h, w, padding) = preprocess(input_img)    
        
    data = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device).float()
    
    with torch.no_grad():
        output = hybrid(data)[0].cpu().numpy().reshape(-1, 2)
        
    output = removePreprocess(output, (h, w, padding))
    
    output = output.astype('int')
    
    outseg = drawOnTop(input_img, output, original_shape) 
    
    seg_to_save = (outseg.copy() * 255).astype('uint8')
    cv2.imwrite("tmp/overlap_segmentation.png" , cv2.cvtColor(seg_to_save, cv2.COLOR_RGB2BGR))
    
    RL = output[0:44]
    LL = output[44:94]
    H = output[94:]
            
    np.savetxt("tmp/RL_landmarks.txt", RL, delimiter=" ", fmt="%d")
    np.savetxt("tmp/LL_landmarks.txt", LL, delimiter=" ", fmt="%d")
    np.savetxt("tmp/H_landmarks.txt", H, delimiter=" ", fmt="%d")
    
    RL_mask, LL_mask, H_mask = getMasks(output, original_shape[0], original_shape[1])
    
    cv2.imwrite("tmp/RL_mask.png", RL_mask)
    cv2.imwrite("tmp/LL_mask.png", LL_mask)
    cv2.imwrite("tmp/H_mask.png", H_mask)
    
    zip = zip_files(["tmp/RL_landmarks.txt", "tmp/LL_landmarks.txt", "tmp/H_landmarks.txt", "tmp/RL_mask.png", "tmp/LL_mask.png", "tmp/H_mask.png", "tmp/overlap_segmentation.png"])    
    
    return outseg, ["tmp/RL_landmarks.txt", "tmp/LL_landmarks.txt", "tmp/H_landmarks.txt", "tmp/RL_mask.png", "tmp/LL_mask.png", "tmp/H_mask.png", "tmp/overlap_segmentation.png", zip]

if __name__ == "__main__":    
    
    with gr.Blocks() as demo:

        gr.Markdown("""
                    # Chest X-ray HybridGNet Segmentation.
                    
                    Demo of the HybridGNet model introduced in "Improving anatomical plausibility in medical image segmentation via hybrid graph neural networks: applications to chest x-ray analysis."
                    
                    Instructions:
                    1. Upload a chest X-ray image (PA or AP) in PNG or JPEG format.
                    2. Click on "Segment Image".
                    
                    Note: Pre-processing is not needed, it will be done automatically and removed after the segmentation.
                    
                    Please check citations below.                    
                    """)

        with gr.Tab("Segment Image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="filepath", height=750)
                    
                    with gr.Row():
                        clear_button = gr.Button("Clear")
                        image_button = gr.Button("Segment Image")
                        
                    gr.Examples(inputs=image_input, examples=['utils/example1.jpg','utils/example2.jpg','utils/example3.png','utils/example4.jpg'])
                        
                with gr.Column():
                    image_output = gr.Image(type="filepath", height=750)
                    results = gr.File()       
        
        gr.Markdown("""
                    If you use this code, please cite:
                    
                    ```
                    @article{gaggion2022TMI,
                        doi = {10.1109/tmi.2022.3224660},
                        url = {https://doi.org/10.1109%2Ftmi.2022.3224660},
                        year = 2022,
                        publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
                        author = {Nicolas Gaggion and Lucas Mansilla and Candelaria Mosquera and Diego H. Milone and Enzo Ferrante},
                        title = {Improving anatomical plausibility in medical image segmentation via hybrid graph neural networks: applications to chest x-ray analysis},
                        journal = {{IEEE} Transactions on Medical Imaging}
                    }
                    ```
                    
                    This model was trained following the procedure explained on:
                    
                    ```
                    @INPROCEEDINGS{gaggion2022ISBI,
                        author={Gaggion, Nicolás and Vakalopoulou, Maria and Milone, Diego H. and Ferrante, Enzo},
                        booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)}, 
                        title={Multi-Center Anatomical Segmentation with Heterogeneous Labels Via Landmark-Based Models}, 
                        year={2023},
                        volume={},
                        number={},
                        pages={1-5},
                        doi={10.1109/ISBI53787.2023.10230691}
                    }
                    ```

                    Example images extracted from Wikipedia, released under:
                    1. CC0 Universial Public Domain. Source: https://commons.wikimedia.org/wiki/File:Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg
                    2. Creative Commons Attribution-Share Alike 4.0 International. Source: https://commons.wikimedia.org/wiki/File:Chest_X-ray.jpg
                    3. Creative Commons Attribution 3.0 Unported. Source https://commons.wikimedia.org/wiki/File:Implantable_cardioverter_defibrillator_chest_X-ray.jpg
                    4. Creative Commons Attribution-Share Alike 3.0 Unported. Source: https://commons.wikimedia.org/wiki/File:Medical_X-Ray_imaging_PRD06_nevit.jpg
                    
                    Author: Nicolás Gaggion
                    Website: [ngaggion.github.io](https://ngaggion.github.io/)
                    
                    """)
        

        clear_button.click(lambda: None, None, image_input, queue=False)
        clear_button.click(lambda: None, None, image_output, queue=False)
        
        image_button.click(segment, inputs=image_input, outputs=[image_output, results], queue=False)
        
    demo.launch()
