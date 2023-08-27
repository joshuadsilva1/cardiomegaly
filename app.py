import numpy as np
import gradio as gr
import cv2 

from models.HybridGNet2IGSC import Hybrid 
from utils.utils import scipy_to_torch_sparse, genMatrixesLungsHeart
import scipy.sparse as sp
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hybrid = None

def getDenseMask(landmarks):
    RL = landmarks[0:44]
    LL = landmarks[44:94]
    H = landmarks[94:]
    
    img = np.zeros([1024,1024], dtype = 'uint8')
    
    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')
    H = H.reshape(-1, 1, 2).astype('int')

    img = cv2.drawContours(img, [RL], -1, 1, -1)
    img = cv2.drawContours(img, [LL], -1, 1, -1)
    img = cv2.drawContours(img, [H], -1, 2, -1)
    
    return img


def drawOnTop(img, landmarks):
    output = getDenseMask(landmarks)
    
    image = np.zeros([1024, 1024, 3])
    image[:,:,0] = img + 0.3 * (output == 1).astype('float') - 0.1 * (output == 2).astype('float')
    image[:,:,1] = img + 0.3 * (output == 2).astype('float') - 0.1 * (output == 1).astype('float') 
    image[:,:,2] = img - 0.1 * (output == 1).astype('float') - 0.2 * (output == 2).astype('float') 

    image = np.clip(image, 0, 1)
    
    RL, LL, H = landmarks[0:44], landmarks[44:94], landmarks[94:]
    
    # Draw the landmarks as dots
    
    for l in RL:
        image = cv2.circle(image, (int(l[0]), int(l[1])), 1, (1, 1, 0), -1)
    for l in LL:
        image = cv2.circle(image, (int(l[0]), int(l[1])), 1, (1, 1, 0), -1)
    for l in H:
        image = cv2.circle(image, (int(l[0]), int(l[1])), 1, (0, 1, 1), -1)
    
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

    
def segment(input_img):
    global hybrid
    
    if hybrid is None:
        hybrid = loadModel()
    
    input_img = cv2.imread(input_img, 0) / 255.0
    
    img, (h, w, padding) = preprocess(input_img)    
        
    data = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device).float()
    
    with torch.no_grad():
        output = hybrid(data)[0].cpu().numpy().reshape(-1, 2) * 1024
       
    return drawOnTop(img, output)


if __name__ == "__main__":    
    demo = gr.Interface(segment, gr.Image(type="filepath"), "image")
    demo.launch()
