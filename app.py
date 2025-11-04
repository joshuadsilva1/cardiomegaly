# import os
# import cv2
# import numpy as np
# import torch
# import gradio as gr
# from pathlib import Path
# import pandas as pd
# import glob 
# import scipy.sparse as sp
# from zipfile import ZipFile
# import tensorflow as tf 
# from tensorflow.keras.models import load_model 

# # -------------------- CONFIG --------------------
# HYBRID_WEIGHTS = "/Users/joshua/College/Chest-x-ray-HybridGNet-Segmentation/weights/weights.pt"
# SEG_OUTPUT_DIR = "segmentation_results"
# MODEL_PATH = "/Users/joshua/College/Chest-x-ray-HybridGNet-Segmentation/models/classifier_best.keras"

# WEIGHTS_PATH = HYBRID_WEIGHTS
# IMG_HEIGHT = 224
# IMG_WIDTH = 224
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# OUTPUT_DIR = SEG_OUTPUT_DIR
# # -------------------- LOAD MODELS --------------------
# hybrid_model = None
# classifier_model = None
# hybrid = None # Initialize the module-level 'hybrid' variable for segmentation

# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.conv.cheb_conv import ChebConv
# from torch_geometric.nn.inits import zeros, normal

# # We change the default initialization from zeros to a normal distribution
# class ChebConv(ChebConv):
#     def reset_parameters(self):
#         for lin in self.lins:
#             normal(lin, mean = 0, std = 0.1)
#             #lin.reset_parameters()
#         normal(self.bias, mean = 0, std = 0.1)
#         #zeros(self.bias)

# # Pooling from COMA: https://github.com/pixelite1201/pytorch_coma/blob/master/layers.py
# class Pool(MessagePassing):
#     def __init__(self):
#         # source_to_target is the default value for flow, but is specified here for explicitness
#         super(Pool, self).__init__(flow='source_to_target')

#     def forward(self, x, pool_mat,  dtype=None):
#         pool_mat = pool_mat.transpose(0, 1)
#         out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(), size=pool_mat.size())
#         return out

#     def message(self, x_j, norm):
#         return norm.view(1, -1, 1) * x_j
    
    
# import torch.nn as nn
# import torch.nn.functional as F

# class residualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         """
#         Args:
#           in_channels (int):  Number of input channels.
#           out_channels (int): Number of output channels.
#           stride (int):       Controls the stride.
#         """
#         super(residualBlock, self).__init__()

#         self.skip = nn.Sequential()

#         if stride != 1 or in_channels != out_channels:
#           self.skip = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
#             nn.BatchNorm2d(out_channels, track_running_stats=False))
#         else:
#           self.skip = None

#         self.block = nn.Sequential(nn.BatchNorm2d(in_channels, track_running_stats=False),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
#                                    nn.BatchNorm2d(out_channels, track_running_stats=False),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(out_channels, out_channels, 3, padding=1)
#                                    )   

#     def forward(self, x):
#         identity = x
#         out = self.block(x)

#         if self.skip is not None:
#             identity = self.skip(x)

#         out += identity
#         out = F.relu(out)

#         return out

# import torchvision.ops.roi_align as roi_align

    
# class EncoderConv(nn.Module):
#     def __init__(self, latents = 64, hw = 32):
#         super(EncoderConv, self).__init__()
        
#         self.latents = latents
#         self.c = 4
        
#         self.size = self.c * np.array([2,4,8,16,32], dtype = np.intc)
        
#         self.maxpool = nn.MaxPool2d(2)
        
#         self.dconv_down1 = residualBlock(1, self.size[0])
#         self.dconv_down2 = residualBlock(self.size[0], self.size[1])
#         self.dconv_down3 = residualBlock(self.size[1], self.size[2])
#         self.dconv_down4 = residualBlock(self.size[2], self.size[3])
#         self.dconv_down5 = residualBlock(self.size[3], self.size[4])
#         self.dconv_down6 = residualBlock(self.size[4], self.size[4])
        
#         self.fc_mu = nn.Linear(in_features=self.size[4]*hw*hw, out_features=self.latents)
#         self.fc_logvar = nn.Linear(in_features=self.size[4]*hw*hw, out_features=self.latents)

#     def forward(self, x):
#         x = self.dconv_down1(x)
#         x = self.maxpool(x)

#         x = self.dconv_down2(x)
#         x = self.maxpool(x)
        
#         conv3 = self.dconv_down3(x)
#         x = self.maxpool(conv3)
        
#         conv4 = self.dconv_down4(x)
#         x = self.maxpool(conv4)
        
#         conv5 = self.dconv_down5(x)
#         x = self.maxpool(conv5)
        
#         conv6 = self.dconv_down6(x)
        
#         x = conv6.view(conv6.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        
#         x_mu = self.fc_mu(x)
#         x_logvar = self.fc_logvar(x)
                
#         return x_mu, x_logvar, conv6, conv5


# class SkipBlock(nn.Module):
#     def __init__(self, in_filters, window):
#         super(SkipBlock, self).__init__()
        
#         self.window = window
#         self.graphConv_pre = ChebConv(in_filters, 2, 1, bias = False) 
    
#     def lookup(self, pos, layer, salida = (1,1)):
#         B = pos.shape[0]
#         N = pos.shape[1]
#         F = layer.shape[1]
#         h = layer.shape[-1]
        
#         ## Scale from [0,1] to [0, h]
#         pos = pos * h
        
#         _x1 = (self.window[0] // 2) * 1.0
#         _x2 = (self.window[0] // 2 + 1) * 1.0
#         _y1 = (self.window[1] // 2) * 1.0
#         _y2 = (self.window[1] // 2 + 1) * 1.0
        
#         boxes = []
#         for batch in range(0, B):
#             x1 = pos[batch,:,0].reshape(-1, 1) - _x1
#             x2 = pos[batch,:,0].reshape(-1, 1) + _x2
#             y1 = pos[batch,:,1].reshape(-1, 1) - _y1
#             y2 = pos[batch,:,1].reshape(-1, 1) + _y2
            
#             aux = torch.cat([x1, y1, x2, y2], axis = 1)            
#             boxes.append(aux)
                    
#         skip = roi_align(layer, boxes, output_size = salida, aligned=True)
#         vista = skip.view([B, N, -1])

#         return vista
    
#     def forward(self, x, adj, conv_layer):
#         pos = self.graphConv_pre(x, adj)
#         skip = self.lookup(pos, conv_layer)
        
#         return torch.cat((x, skip, pos), axis = 2), pos
        
    
# class Hybrid(nn.Module):
#     def __init__(self, config, downsample_matrices, upsample_matrices, adjacency_matrices):
#         super(Hybrid, self).__init__()
        
#         self.config = config
#         hw = config['inputsize'] // 32
#         self.z = config['latents']
#         self.encoder = EncoderConv(latents = self.z, hw = hw)
        
#         self.downsample_matrices = downsample_matrices
#         self.upsample_matrices = upsample_matrices
#         self.adjacency_matrices = adjacency_matrices
#         self.kld_weight = 1e-5
                
#         n_nodes = config['n_nodes']
#         self.filters = config['filters']
#         self.K = 6
#         self.window = (3,3)
        
#         # Genero la capa fully connected del decoder
#         outshape = self.filters[-1] * n_nodes[-1]          
#         self.dec_lin = torch.nn.Linear(self.z, outshape)
                
#         self.normalization2u = torch.nn.InstanceNorm1d(self.filters[1])
#         self.normalization3u = torch.nn.InstanceNorm1d(self.filters[2])
#         self.normalization4u = torch.nn.InstanceNorm1d(self.filters[3])
#         self.normalization5u = torch.nn.InstanceNorm1d(self.filters[4])
#         self.normalization6u = torch.nn.InstanceNorm1d(self.filters[5])
        
#         outsize1 = self.encoder.size[4]
#         outsize2 = self.encoder.size[4]  
                     
#         # Guardo las capas de convoluciones en grafo
#         self.graphConv_up6 = ChebConv(self.filters[6], self.filters[5], self.K)
#         self.graphConv_up5 = ChebConv(self.filters[5], self.filters[4], self.K)       
        
#         self.SC_1 = SkipBlock(self.filters[4], self.window)
        
#         self.graphConv_up4 = ChebConv(self.filters[4] + outsize1 + 2, self.filters[3], self.K)        
#         self.graphConv_up3 = ChebConv(self.filters[3], self.filters[2], self.K)
        
#         self.SC_2 = SkipBlock(self.filters[2], self.window)
        
#         self.graphConv_up2 = ChebConv(self.filters[2] + outsize2 + 2, self.filters[1], self.K)
#         self.graphConv_up1 = ChebConv(self.filters[1], self.filters[0], 1, bias = False)
                
#         self.pool = Pool()
        
#         self.reset_parameters()
        
#     def reset_parameters(self):
#         torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)


#     def sampling(self, mu, log_var):
#         std = torch.exp(0.5*log_var)
#         eps = torch.randn_like(std)
#         return eps.mul(std).add_(mu) 
    
        
#     def forward(self, x):
#         self.mu, self.log_var, conv6, conv5 = self.encoder(x)

#         if self.training:
#             z = self.sampling(self.mu, self.log_var)
#         else:
#             z = self.mu
            
#         x = self.dec_lin(z)
#         x = F.relu(x)
        
#         x = x.reshape(x.shape[0], -1, self.filters[-1])
        
#         x = self.graphConv_up6(x, self.adjacency_matrices[5]._indices())
#         x = self.normalization6u(x)
#         x = F.relu(x)
        
#         x = self.graphConv_up5(x, self.adjacency_matrices[4]._indices())
#         x = self.normalization5u(x)
#         x = F.relu(x)
        
#         x, pos1 = self.SC_1(x, self.adjacency_matrices[3]._indices(), conv6)
        
#         x = self.graphConv_up4(x, self.adjacency_matrices[3]._indices())
#         x = self.normalization4u(x)
#         x = F.relu(x)
        
#         x = self.pool(x, self.upsample_matrices[0])
        
#         x = self.graphConv_up3(x, self.adjacency_matrices[2]._indices())
#         x = self.normalization3u(x)
#         x = F.relu(x)
        
#         x, pos2 = self.SC_2(x, self.adjacency_matrices[1]._indices(), conv5)
        
#         x = self.graphConv_up2(x, self.adjacency_matrices[1]._indices())
#         x = self.normalization2u(x)
#         x = F.relu(x)
        
#         x = self.graphConv_up1(x, self.adjacency_matrices[0]._indices()) # Sin relu y sin bias
        
#         return x, pos1, pos2

# def scipy_to_torch_sparse(scp_matrix):
#     values = scp_matrix.data
#     indices = np.vstack((scp_matrix.row, scp_matrix.col))
#     i = torch.LongTensor(indices)
#     v = torch.FloatTensor(values)
#     shape = scp_matrix.shape

#     sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
#     return sparse_tensor

# ## Adjacency Matrix
# def mOrgan(N):
#     sub = np.zeros([N, N])
#     for i in range(0, N):
#         sub[i, i-1] = 1
#         sub[i, (i+1)%N] = 1
#     return sub

# ## Downsampling Matrix
# def mOrganD(N):
#     N2 = int(np.ceil(N/2))
#     sub = np.zeros([N2, N])
    
#     for i in range(0, N2):
#         if (2*i+1) == N:
#             sub[i, 2*i] = 1
#         else:
#             sub[i, 2*i] = 1/2
#             sub[i, 2*i+1] = 1/2
            
#     return sub

# def mOrganU(N):
#     N2 = int(np.ceil(N/2))
#     sub = np.zeros([N, N2])
    
#     for i in range(0, N):
#         if i % 2 == 0:
#             sub[i, i//2] = 1
#         else:
#             sub[i, i//2] = 1/2
#             sub[i, (i//2 + 1) % N2] = 1/2
            
#     return sub

# def genMatrixesLungsHeart():       
#     RLUNG = 44
#     LLUNG = 50
#     HEART = 26
    
#     Asub1 = mOrgan(RLUNG)
#     Asub2 = mOrgan(LLUNG)
#     Asub3 = mOrgan(HEART)
    
#     ADsub1 = mOrgan(int(np.ceil(RLUNG / 2)))
#     ADsub2 = mOrgan(int(np.ceil(LLUNG / 2)))
#     ADsub3 = mOrgan(int(np.ceil(HEART / 2)))
                    
#     Dsub1 = mOrganD(RLUNG)
#     Dsub2 = mOrganD(LLUNG)
#     Dsub3 = mOrganD(HEART)
    
#     Usub1 = mOrganU(RLUNG)
#     Usub2 = mOrganU(LLUNG)
#     Usub3 = mOrganU(HEART)
        
#     p1 = RLUNG
#     p2 = p1 + LLUNG
#     p3 = p2 + HEART
    
#     p1_ = int(np.ceil(RLUNG / 2))
#     p2_ = p1_ + int(np.ceil(LLUNG / 2))
#     p3_ = p2_ + int(np.ceil(HEART / 2))
    
#     A = np.zeros([p3, p3])
    
#     A[:p1, :p1] = Asub1
#     A[p1:p2, p1:p2] = Asub2
#     A[p2:p3, p2:p3] = Asub3
    
#     AD = np.zeros([p3_, p3_])
    
#     AD[:p1_, :p1_] = ADsub1
#     AD[p1_:p2_, p1_:p2_] = ADsub2
#     AD[p2_:p3_, p2_:p3_] = ADsub3
   
#     D = np.zeros([p3_, p3])
    
#     D[:p1_, :p1] = Dsub1
#     D[p1_:p2_, p1:p2] = Dsub2
#     D[p2_:p3_, p2:p3] = Dsub3
    
#     U = np.zeros([p3, p3_])
    
#     U[:p1, :p1_] = Usub1
#     U[p1:p2, p1_:p2_] = Usub2
#     U[p2:p3, p2_:p3_] = Usub3
    
#     return A, AD, D, U



# def getDenseMask(landmarks, h, w):
#     # ... (function body from app.py)
#     RL = landmarks[0:44]
#     LL = landmarks[44:94]
#     H = landmarks[94:]
    
#     img = np.zeros([h, w], dtype = 'uint8')
    
#     RL = RL.reshape(-1, 1, 2).astype('int')
#     LL = LL.reshape(-1, 1, 2).astype('int')
#     H = H.reshape(-1, 1, 2).astype('int')

#     img = cv2.drawContours(img, [RL], -1, 1, -1)
#     img = cv2.drawContours(img, [LL], -1, 1, -1)
#     img = cv2.drawContours(img, [H], -1, 2, -1)
    
#     return img

# def getMasks(landmarks, h, w):
#     # ... (function body from app.py)
#     RL = landmarks[0:44]
#     LL = landmarks[44:94]
#     H = landmarks[94:]
    
#     RL = RL.reshape(-1, 1, 2).astype('int')
#     LL = LL.reshape(-1, 1, 2).astype('int')
#     H = H.reshape(-1, 1, 2).astype('int')
    
#     RL_mask = np.zeros([h, w], dtype = 'uint8')
#     LL_mask = np.zeros([h, w], dtype = 'uint8')
#     H_mask = np.zeros([h, w], dtype = 'uint8')
    
#     RL_mask = cv2.drawContours(RL_mask, [RL], -1, 255, -1)
#     LL_mask = cv2.drawContours(LL_mask, [LL], -1, 255, -1)
#     H_mask = cv2.drawContours(H_mask, [H], -1, 255, -1)

#     return RL_mask, LL_mask, H_mask

# def drawOnTop(img, landmarks, original_shape):
#     # ... (function body from app.py)
#     h, w = original_shape
#     output = getDenseMask(landmarks, h, w)
    
#     image = np.zeros([h, w, 3])
#     image[:,:,0] = img + 0.3 * (output == 1).astype('float') - 0.1 * (output == 2).astype('float')
#     image[:,:,1] = img + 0.3 * (output == 2).astype('float') - 0.1 * (output == 1).astype('float') 
#     image[:,:,2] = img - 0.1 * (output == 1).astype('float') - 0.2 * (output == 2).astype('float') 

#     image = np.clip(image, 0, 1)
    
#     RL, LL, H = landmarks[0:44], landmarks[44:94], landmarks[94:]
    
#     # Draw the landmarks as dots
    
#     for l in RL:
#         image = cv2.circle(image, (int(l[0]), int(l[1])), 5, (1, 0, 1), -1)
#     for l in LL:
#         image = cv2.circle(image, (int(l[0]), int(l[1])), 5, (1, 0, 1), -1)
#     for l in H:
#         image = cv2.circle(image, (int(l[0]), int(l[1])), 5, (1, 1, 0), -1)
    
#     return image
    

# def loadModel(device):    
#     # ... (function body from app.py)
#     A, AD, D, U = genMatrixesLungsHeart()
#     N1 = A.shape[0]
#     N2 = AD.shape[0]

#     A = sp.csc_matrix(A).tocoo()
#     AD = sp.csc_matrix(AD).tocoo()
#     D = sp.csc_matrix(D).tocoo()
#     U = sp.csc_matrix(U).tocoo()

#     D_ = [D.copy()]
#     U_ = [U.copy()]

#     config = {}

#     config['n_nodes'] = [N1, N1, N1, N2, N2, N2]
#     A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]
    
#     A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to(device) for x in X] for X in (A_, D_, U_))

#     config['latents'] = 64
#     config['inputsize'] = 1024

#     f = 32
#     config['filters'] = [2, f, f, f, f//2, f//2, f//2]
#     config['skip_features'] = f

#     hybrid = Hybrid(config.copy(), D_t, U_t, A_t).to(device)
#     hybrid.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device(device)))
#     hybrid.eval()
    
#     return hybrid


# def pad_to_square(img):
#     # ... (function body from app.py)
#     h, w = img.shape[:2]
    
#     if h > w:
#         padw = (h - w) 
#         auxw = padw % 2
#         img = np.pad(img, ((0, 0), (padw//2, padw//2 + auxw)), 'constant')
        
#         padh = 0
#         auxh = 0
        
#     else:
#         padh = (w - h) 
#         auxh = padh % 2
#         img = np.pad(img, ((padh//2, padh//2 + auxh), (0, 0)), 'constant')

#         padw = 0
#         auxw = 0
        
#     return img, (padh, padw, auxh, auxw)
    

# def preprocess(input_img):
#     # ... (function body from app.py)
#     img, padding = pad_to_square(input_img)
    
#     h, w = img.shape[:2]
#     if h != 1024 or w != 1024:
#         img = cv2.resize(img, (1024, 1024), interpolation = cv2.INTER_CUBIC)
        
#     return img, (h, w, padding)


# def removePreprocess(output, info):
#     # ... (function body from app.py)
#     h, w, padding = info
    
#     if h != 1024 or w != 1024:
#         output = output * h
#     else:
#         output = output * 1024
    
#     padh, padw, auxh, auxw = padding
    
#     output[:, 0] = output[:, 0] - padw//2
#     output[:, 1] = output[:, 1] - padh//2
    
#     return output   


# def zip_files(files, zip_filename):
#     # The zip function is modified to take a custom filename
#     zip_path = os.path.join(OUTPUT_DIR, zip_filename)
#     with ZipFile(zip_path, "w") as zipObj:
#         for idx, file in enumerate(files):
#             # Ensure the archive name is only the base filename, not the full path
#             zipObj.write(file, arcname=os.path.basename(file)) 
#     return zip_path


# def segment(input_img_path):
#     # The segment function is modified to create a unique sub-directory 
#     # for each image's results.
    
#     global hybrid, DEVICE # <-- FIX APPLIED HERE
    
#     # 1. Check/Load Model
#     if hybrid is None:
#         print("Loading HybridGNet model...")
#         hybrid = loadModel(DEVICE)
#         print("Model loaded.")
    
#     # 2. Setup output directory for this specific image
#     base_name = os.path.splitext(os.path.basename(input_img_path))[0]
#     local_output_dir = os.path.join(OUTPUT_DIR, base_name)
#     os.makedirs(local_output_dir, exist_ok=True)
    
    
#     # 3. Preprocessing
#     input_img = cv2.imread(input_img_path, 0) / 255.0
#     if input_img is None:
#         raise FileNotFoundError(f"Could not load image at {input_img_path}")
        
#     original_shape = input_img.shape[:2]
    
#     img, (h, w, padding) = preprocess(input_img)    
        
#     data = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE).float()
    
#     # 4. Inference
#     with torch.no_grad():
#         output = hybrid(data)[0].cpu().numpy().reshape(-1, 2)
        
#     output = removePreprocess(output, (h, w, padding))
#     output = output.astype('int')
    
#     # 5. Save Outputs
    
#     # Define paths within the specific image's output folder
    
#     # Overlap Segmentation Image
#     outseg = drawOnTop(input_img, output, original_shape) 
#     seg_to_save = (outseg.copy() * 255).astype('uint8')
#     overlap_path = os.path.join(local_output_dir, f"{base_name}_overlap_segmentation.png")
#     cv2.imwrite(overlap_path , cv2.cvtColor(seg_to_save, cv2.COLOR_RGB2BGR))
    
#     RL = output[0:44]
#     LL = output[44:94]
#     H = output[94:]
    
#     # Landmark Text Files
#     rl_path = os.path.join(local_output_dir, f"{base_name}_RL_landmarks.txt")
#     ll_path = os.path.join(local_output_dir, f"{base_name}_LL_landmarks.txt")
#     h_path = os.path.join(local_output_dir, f"{base_name}_H_landmarks.txt")
    
#     np.savetxt(rl_path, RL, delimiter=" ", fmt="%d")
#     np.savetxt(ll_path, LL, delimiter=" ", fmt="%d")
#     np.savetxt(h_path, H, delimiter=" ", fmt="%d")
    
#     # Binary Mask Images
#     RL_mask, LL_mask, H_mask = getMasks(output, original_shape[0], original_shape[1])
    
#     rl_mask_path = os.path.join(local_output_dir, f"{base_name}_RL_mask.png")
#     ll_mask_path = os.path.join(local_output_dir, f"{base_name}_LL_mask.png")
#     h_mask_path = os.path.join(local_output_dir, f"{base_name}_H_mask.png")
    
#     cv2.imwrite(rl_mask_path, RL_mask)
#     cv2.imwrite(ll_mask_path, LL_mask)
#     cv2.imwrite(h_mask_path, H_mask)
    
#     return local_output_dir


# def batch_segmentation(input_dir):
#     """
#     Iterates through all supported image files in a directory and segments them.
#     """
    
#     if not os.path.isdir(input_dir):
#         print(f"ERROR: Input directory not found: {input_dir}")
#         return

#     # Supported image extensions
#     extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
#     image_paths = []
#     for ext in extensions:
#         # glob.glob is used to find all files matching the pattern (e.g., all *.jpg files)
#         image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

#     if not image_paths:
#         print(f"No supported images found in the directory: {input_dir}")
#         return

#     print(f"Found {len(image_paths)} images to process.")

#     # Process each image
#     for path in image_paths:
#         try:
#             segment(path)
#         except FileNotFoundError as e:
#             print(f"Skipping {os.path.basename(path)}: {e}")
#         except Exception as e:
#             print(f"An unexpected error occurred while processing {os.path.basename(path)}: {e}")
    
#     print("\n--- Batch Segmentation Complete ---")



# def load_hybrid():
#     global hybrid_model
#     if hybrid_model is None:
#         print("Loading HybridGNet model...")
#         hybrid_model = loadModel(DEVICE)  # your function from earlier
#         print("HybridGNet loaded.")
#     return hybrid_model

# def load_classifier():
#     global classifier_model
#     if classifier_model is None:
#         # NOTE: The load_model import is now at the top of the script
#         classifier_model = load_model(MODEL_PATH, compile=False)
#         print("Classifier loaded.")
#     return classifier_model

# # -------------------- HELPER FUNCTIONS --------------------
# def calculate_mask_features(mask): 
#     """
#     Calculates the fractional area of heart and lungs from the mask.
#     Mask convention: 1 for lung, 2 for heart.
#     """
#     # FIX: Ensure mask is an array before calculation
#     mask = np.asarray(mask)
#     heart_frac = np.mean(mask == 2) 
#     lungs_frac = np.mean(mask == 1)
#     return heart_frac, lungs_frac


# def calculate_cdr_from_landmarks(rl_data, ll_data, h_data):
#     """
#     Calculates the Cardiopulmonary Ratio (CDR) from landmark data.
#     """
    
#     # --- 1. Helper function to parse data ---
#     def parse_landmarks(data):
#         """Parses the content string into a list of (Y, X) coordinates."""
#         coords = []
#         # Split lines, skip the header/source line
#         lines = data.strip().split('\n')
#         for line in lines:
#             if not line.strip():
#                 continue
#             try:
#                 # The segment function saves 'X Y', so read as 'X Y'
#                 x, y = map(int, line.strip().split())
#                 coords.append((y, x)) # Store as (Y, X) for easier indexing of rows/cols
#             except ValueError:
#                 # Skip any lines that don't conform to 'int int' or empty lines
#                 continue
#         return coords

#     # Load landmark data from the saved text files
#     try:
#         with open(rl_data, 'r') as f: rl_coords = parse_landmarks(f.read())
#         with open(ll_data, 'r') as f: ll_coords = parse_landmarks(f.read())
#         with open(h_data, 'r') as f: h_coords = parse_landmarks(f.read())
#     except FileNotFoundError:
#         return None

#     if not rl_coords or not ll_coords or not h_coords:
#         return None # Cannot calculate if data is missing

#     # Extract X (column/width) coordinates
#     x_rl = [x for y, x in rl_coords]
#     x_ll = [x for y, x in ll_coords]
#     x_h = [x for y, x in h_coords]


#     # --- 2. Calculate Thoracic Width (Max width of both lungs) ---
#     x_thorax_max = max(x_ll) # X-coordinate of the rightmost point of the Left Lung
#     x_thorax_min = min(x_rl) # X-coordinate of the leftmost point of the Right Lung
#     thoracic_width = x_thorax_max - x_thorax_min

#     # --- 3. Calculate Heart Width ---
#     heart_width = max(x_h) - min(x_h)

#     # --- 4. Calculate CDR ---
#     if thoracic_width <= 0:
#         return None # Avoid division by zero
        
#     cdr = heart_width / thoracic_width
#     return cdr


# def build_combined_mask_from_seg_folder(base, seg_folder, target_size):
#     """
#     Combines individual saved mask files into a single mask array.
#     """
    
#     # Read the saved mask files
#     rl_mask_path = os.path.join(seg_folder, f"{base}_RL_mask.png")
#     ll_mask_path = os.path.join(seg_folder, f"{base}_LL_mask.png")
#     h_mask_path = os.path.join(seg_folder, f"{base}_H_mask.png")
    
#     RL_mask = cv2.imread(rl_mask_path, 0)
#     LL_mask = cv2.imread(ll_mask_path, 0)
#     H_mask = cv2.imread(h_mask_path, 0)
    
#     if RL_mask is None or LL_mask is None or H_mask is None:
#         # Try to continue if masks are missing, but this is a serious error
#         raise FileNotFoundError("One or more segmentation masks were not saved correctly.")
    
#     # The convention from getDenseMask: 1 for lungs, 2 for heart
#     combined_orig = np.zeros_like(RL_mask, dtype=np.uint8)
    
#     # Lungs (value 1)
#     combined_orig[RL_mask > 0] = 1
#     combined_orig[LL_mask > 0] = 1
    
#     # Heart (value 2). Heart overlaps lungs, so set it last.
#     combined_orig[H_mask > 0] = 2
    
#     # Resize to the target size for the classifier input
#     combined_resized = cv2.resize(combined_orig.astype(np.float32), 
#                                   target_size, 
#                                   interpolation=cv2.INTER_NEAREST)
    
#     return combined_orig, combined_resized.astype(np.uint8)


# def compute_ctr(mask_folder, base):
#     """
#     Calculates the Cardiothoracic Ratio by reading landmark data from files.
#     """
#     rl_path = os.path.join(mask_folder, f"{base}_RL_landmarks.txt")
#     ll_path = os.path.join(mask_folder, f"{base}_LL_landmarks.txt")
#     h_path = os.path.join(mask_folder, f"{base}_H_landmarks.txt")
    
#     cdr = calculate_cdr_from_landmarks(rl_path, ll_path, h_path)
#     return cdr if cdr is not None else 0.5 # Default to 0.5 if calculation fails


# # -------------------- INFERENCE FUNCTION --------------------
# def predict_cardiomegaly(image_path):
    
#     # 1. Segment image
#     seg_folder = segment(image_path)  # HybridGNet segmentation
#     base = Path(image_path).stem

#     # 2. Build combined mask and compute CDR
#     combined_orig, combined_resized = build_combined_mask_from_seg_folder(base, seg_folder, target_size=(IMG_HEIGHT, IMG_WIDTH))
    
#     # The CTR is calculated from the saved landmark files
#     ctr = compute_ctr(seg_folder, base)
    
#     # 3. Compute features for classifier input
#     heart_frac, lungs_frac = calculate_mask_features(combined_resized) # Fraction features on the resized mask
    
#     # 4. Prepare inputs for classifier
#     img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255.0
#     img_gray = cv2.resize(img_gray, (IMG_WIDTH, IMG_HEIGHT))
    
#     # The combined_resized mask is 0 (background), 1 (lung), 2 (heart)
#     heart_channel = (combined_resized==2).astype(np.float32) # Heart is 2
#     lungs_channel = (combined_resized==1).astype(np.float32) # Lungs are 1
    
#     img_input = np.stack([img_gray, heart_channel, lungs_channel], axis=-1)
#     img_input = np.expand_dims(img_input, axis=0)  # batch dimension
    
#     num_input = np.array([[ctr, heart_frac, lungs_frac]], dtype=np.float32)
    
#     # 5. Predict
#     classifier = load_classifier()
#     prob = classifier.predict([img_input, num_input])[0][0]
    
#     # 6. Format Outputs
    
#     # Load all image outputs
#     overlap_path = os.path.join(seg_folder, f"{base}_overlap_segmentation.png")
#     rl_mask_path = os.path.join(seg_folder, f"{base}_RL_mask.png")
#     ll_mask_path = os.path.join(seg_folder, f"{base}_LL_mask.png")
#     h_mask_path = os.path.join(seg_folder, f"{base}_H_mask.png")
    
#     overlap_img = cv2.cvtColor(cv2.imread(overlap_path), cv2.COLOR_BGR2RGB)
#     rl_mask_img = cv2.imread(rl_mask_path, 0)
#     ll_mask_img = cv2.imread(ll_mask_path, 0)
#     h_mask_img = cv2.imread(h_mask_path, 0)
    
#     # Determine the final diagnosis label
#     likelihood = prob * 100
#     if likelihood > 50.0:
#         diagnosis = f"Susceptible to Cardiomegaly ({likelihood:.2f}%)"
#     else:
#         diagnosis = f"Likely No Cardiomegaly ({likelihood:.2f}%)"
        
#     # FIX: Change the output format to a dictionary suitable for gr.JSON
#     output_data = {
#         "Cardiothoracic Ratio (CDR)": f"{ctr:.4f}",
#         "Prediction Likelihood": f"{likelihood:.2f}%",
#         "Final Assessment": diagnosis
#     }

#     # FIX: Return the structured data
#     return overlap_img, rl_mask_img, ll_mask_img, h_mask_img, output_data


# # -------------------- GRADIO INTERFACE --------------------
# # FIX: Change the last output component from gr.Label to gr.JSON
# iface = gr.Interface(
#     fn=predict_cardiomegaly,
#     inputs=gr.File(file_types=[".png", ".jpg", ".jpeg"]),
#     outputs=[
#         gr.Image(type="numpy", label="Overlap Segmentation (Heart: Cyan, Lungs: Magenta)"),
#         gr.Image(type="numpy", label="Right Lung Mask"),
#         gr.Image(type="numpy", label="Left Lung Mask"),
#         gr.Image(type="numpy", label="Heart Mask"),
#         gr.JSON(label="Cardiomegaly Prediction & Metrics"), # <--- FIX APPLIED HERE
#     ],
#     title="Chest X-ray Cardiomegaly Prediction using HybridGNet",
#     description="Upload a postero-anterior (PA) chest X-ray image. The model performs joint segmentation and classification to predict the likelihood of cardiomegaly and calculates the Cardiothoracic Ratio (CDR)."
# )

# if __name__ == "__main__":
    
#     # Create the output directory if it doesn't exist
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     iface.launch(server_name="0.0.0.0", server_port=7890, share=True)


# app.py (Main Gradio Client)
import os
# --- CRITICAL FIX: RAG Protection (Keep for stability) ---
os.environ["TRANSFORMERS_NO_TF"] = "1"
# --------------------------------------------------------
import cv2
import numpy as np
import torch
import gradio as gr
from pathlib import Path
import pandas as pd
import glob 
import scipy.sparse as sp
from zipfile import ZipFile
import tensorflow as tf 
from tensorflow.keras.models import load_model 
import requests # Necessary for API communication
import time 
import re # Keep if used by the segment/drawOnTop utility functions

# -------------------- CONFIG --------------------
HYBRID_WEIGHTS = "/Users/joshua/College/Chest-x-ray-HybridGNet-Segmentation/weights/weights.pt"
SEG_OUTPUT_DIR = "segmentation_results"
MODEL_PATH = "/Users/joshua/College/Chest-x-ray-HybridGNet-Segmentation/models/classifier_best.keras"

WEIGHTS_PATH = HYBRID_WEIGHTS
IMG_HEIGHT = 224
IMG_WIDTH = 224
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = SEG_OUTPUT_DIR

# --- API CONFIG ---
RAG_SERVER_URL = "http://127.0.0.1:8000/query" # Address of the external RAG service
# -------------------- LOAD MODELS (Class Definitions Omitted for Brevity) --------------------
hybrid_model = None
classifier_model = None
hybrid = None 

# --- Class Definitions (ChebConv, Pool, residualBlock, EncoderConv, SkipBlock, Hybrid) must be present here ---
# --- Utility Functions (scipy_to_torch_sparse, mOrgan, genMatrixesLungsHeart, etc.) must be present here ---

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.cheb_conv import ChebConv
from torch_geometric.nn.inits import zeros, normal

# We change the default initialization from zeros to a normal distribution
class ChebConv(ChebConv):
    def reset_parameters(self):
        for lin in self.lins:
            normal(lin, mean = 0, std = 0.1)
            #lin.reset_parameters()
        normal(self.bias, mean = 0, std = 0.1)
        #zeros(self.bias)

# Pooling from COMA: https://github.com/pixelite1201/pytorch_coma/blob/master/layers.py
class Pool(MessagePassing):
    def __init__(self):
        # source_to_target is the default value for flow, but is specified here for explicitness
        super(Pool, self).__init__(flow='source_to_target')

    def forward(self, x, pool_mat,  dtype=None):
        pool_mat = pool_mat.transpose(0, 1)
        out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(), size=pool_mat.size())
        return out

    def message(self, x_j, norm):
        return norm.view(1, -1, 1) * x_j
    
    
import torch.nn as nn
import torch.nn.functional as F

class residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(residualBlock, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False))
        else:
          self.skip = None

        self.block = nn.Sequential(nn.BatchNorm2d(in_channels, track_running_stats=False),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                   nn.BatchNorm2d(out_channels, track_running_stats=False),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, 3, padding=1)
                                   )   

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out

import torchvision.ops.roi_align as roi_align

    
class EncoderConv(nn.Module):
    def __init__(self, latents = 64, hw = 32):
        super(EncoderConv, self).__init__()
        
        self.latents = latents
        self.c = 4
        
        self.size = self.c * np.array([2,4,8,16,32], dtype = np.intc)
        
        self.maxpool = nn.MaxPool2d(2)
        
        self.dconv_down1 = residualBlock(1, self.size[0])
        self.dconv_down2 = residualBlock(self.size[0], self.size[1])
        self.dconv_down3 = residualBlock(self.size[1], self.size[2])
        self.dconv_down4 = residualBlock(self.size[2], self.size[3])
        self.dconv_down5 = residualBlock(self.size[3], self.size[4])
        self.dconv_down6 = residualBlock(self.size[4], self.size[4])
        
        self.fc_mu = nn.Linear(in_features=self.size[4]*hw*hw, out_features=self.latents)
        self.fc_logvar = nn.Linear(in_features=self.size[4]*hw*hw, out_features=self.latents)

    def forward(self, x):
        x = self.dconv_down1(x)
        x = self.maxpool(x)

        x = self.dconv_down2(x)
        x = self.maxpool(x)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        
        conv5 = self.dconv_down5(x)
        x = self.maxpool(conv5)
        
        conv6 = self.dconv_down6(x)
        
        x = conv6.view(conv6.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
                
        return x_mu, x_logvar, conv6, conv5


class SkipBlock(nn.Module):
    def __init__(self, in_filters, window):
        super(SkipBlock, self).__init__()
        
        self.window = window
        self.graphConv_pre = ChebConv(in_filters, 2, 1, bias = False) 
    
    def lookup(self, pos, layer, salida = (1,1)):
        B = pos.shape[0]
        N = pos.shape[1]
        F = layer.shape[1]
        h = layer.shape[-1]
        
        ## Scale from [0,1] to [0, h]
        pos = pos * h
        
        _x1 = (self.window[0] // 2) * 1.0
        _x2 = (self.window[0] // 2 + 1) * 1.0
        _y1 = (self.window[1] // 2) * 1.0
        _y2 = (self.window[1] // 2 + 1) * 1.0
        
        boxes = []
        for batch in range(0, B):
            x1 = pos[batch,:,0].reshape(-1, 1) - _x1
            x2 = pos[batch,:,0].reshape(-1, 1) + _x2
            y1 = pos[batch,:,1].reshape(-1, 1) - _y1
            y2 = pos[batch,:,1].reshape(-1, 1) + _y2
            
            aux = torch.cat([x1, y1, x2, y2], axis = 1)            
            boxes.append(aux)
                    
        skip = roi_align(layer, boxes, output_size = salida, aligned=True)
        vista = skip.view([B, N, -1])

        return vista
    
    def forward(self, x, adj, conv_layer):
        pos = self.graphConv_pre(x, adj)
        skip = self.lookup(pos, conv_layer)
        
        return torch.cat((x, skip, pos), axis = 2), pos
        
    
class Hybrid(nn.Module):
    def __init__(self, config, downsample_matrices, upsample_matrices, adjacency_matrices):
        super(Hybrid, self).__init__()
        
        self.config = config
        hw = config['inputsize'] // 32
        self.z = config['latents']
        self.encoder = EncoderConv(latents = self.z, hw = hw)
        
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.kld_weight = 1e-5
                
        n_nodes = config['n_nodes']
        self.filters = config['filters']
        self.K = 6
        self.window = (3,3)
        
        # Genero la capa fully connected del decoder
        outshape = self.filters[-1] * n_nodes[-1]          
        self.dec_lin = torch.nn.Linear(self.z, outshape)
                
        self.normalization2u = torch.nn.InstanceNorm1d(self.filters[1])
        self.normalization3u = torch.nn.InstanceNorm1d(self.filters[2])
        self.normalization4u = torch.nn.InstanceNorm1d(self.filters[3])
        self.normalization5u = torch.nn.InstanceNorm1d(self.filters[4])
        self.normalization6u = torch.nn.InstanceNorm1d(self.filters[5])
        
        outsize1 = self.encoder.size[4]
        outsize2 = self.encoder.size[4]  
                     
        # Guardo las capas de convoluciones en grafo
        self.graphConv_up6 = ChebConv(self.filters[6], self.filters[5], self.K)
        self.graphConv_up5 = ChebConv(self.filters[5], self.filters[4], self.K)       
        
        self.SC_1 = SkipBlock(self.filters[4], self.window)
        
        self.graphConv_up4 = ChebConv(self.filters[4] + outsize1 + 2, self.filters[3], self.K)        
        self.graphConv_up3 = ChebConv(self.filters[3], self.filters[2], self.K)
        
        self.SC_2 = SkipBlock(self.filters[2], self.window)
        
        self.graphConv_up2 = ChebConv(self.filters[2] + outsize2 + 2, self.filters[1], self.K)
        self.graphConv_up1 = ChebConv(self.filters[1], self.filters[0], 1, bias = False)
                
        self.pool = Pool()
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)


    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 
    
        
    def forward(self, x):
        self.mu, self.log_var, conv6, conv5 = self.encoder(x)

        if self.training:
            z = self.sampling(self.mu, self.log_var)
        else:
            z = self.mu
            
        x = self.dec_lin(z)
        x = F.relu(x)
        
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        
        x = self.graphConv_up6(x, self.adjacency_matrices[5]._indices())
        x = self.normalization6u(x)
        x = F.relu(x)
        
        x = self.graphConv_up5(x, self.adjacency_matrices[4]._indices())
        x = self.normalization5u(x)
        x = F.relu(x)
        
        x, pos1 = self.SC_1(x, self.adjacency_matrices[3]._indices(), conv6)
        
        x = self.graphConv_up4(x, self.adjacency_matrices[3]._indices())
        x = self.normalization4u(x)
        x = F.relu(x)
        
        x = self.pool(x, self.upsample_matrices[0])
        
        x = self.graphConv_up3(x, self.adjacency_matrices[2]._indices())
        x = self.normalization3u(x)
        x = F.relu(x)
        
        x, pos2 = self.SC_2(x, self.adjacency_matrices[1]._indices(), conv5)
        
        x = self.graphConv_up2(x, self.adjacency_matrices[1]._indices())
        x = self.normalization2u(x)
        x = F.relu(x)
        
        x = self.graphConv_up1(x, self.adjacency_matrices[0]._indices()) # Sin relu y sin bias
        
        return x, pos1, pos2

def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

## Adjacency Matrix
def mOrgan(N):
    sub = np.zeros([N, N])
    for i in range(0, N):
        sub[i, i-1] = 1
        sub[i, (i+1)%N] = 1
    return sub

## Downsampling Matrix
def mOrganD(N):
    N2 = int(np.ceil(N/2))
    sub = np.zeros([N2, N])
    
    for i in range(0, N2):
        if (2*i+1) == N:
            sub[i, 2*i] = 1
        else:
            sub[i, 2*i] = 1/2
            sub[i, 2*i+1] = 1/2
            
    return sub

def mOrganU(N):
    N2 = int(np.ceil(N/2))
    sub = np.zeros([N, N2])
    
    for i in range(0, N):
        if i % 2 == 0:
            sub[i, i//2] = 1
        else:
            sub[i, i//2] = 1/2
            sub[i, (i//2 + 1) % N2] = 1/2
            
    return sub

def genMatrixesLungsHeart():       
    RLUNG = 44
    LLUNG = 50
    HEART = 26
    
    Asub1 = mOrgan(RLUNG)
    Asub2 = mOrgan(LLUNG)
    Asub3 = mOrgan(HEART)
    
    ADsub1 = mOrgan(int(np.ceil(RLUNG / 2)))
    ADsub2 = mOrgan(int(np.ceil(LLUNG / 2)))
    ADsub3 = mOrgan(int(np.ceil(HEART / 2)))
                    
    Dsub1 = mOrganD(RLUNG)
    Dsub2 = mOrganD(LLUNG)
    Dsub3 = mOrganD(HEART)
    
    Usub1 = mOrganU(RLUNG)
    Usub2 = mOrganU(LLUNG)
    Usub3 = mOrganU(HEART)
        
    p1 = RLUNG
    p2 = p1 + LLUNG
    p3 = p2 + HEART
    
    p1_ = int(np.ceil(RLUNG / 2))
    p2_ = p1_ + int(np.ceil(LLUNG / 2))
    p3_ = p2_ + int(np.ceil(HEART / 2))
    
    A = np.zeros([p3, p3])
    
    A[:p1, :p1] = Asub1
    A[p1:p2, p1:p2] = Asub2
    A[p2:p3, p2:p3] = Asub3
    
    AD = np.zeros([p3_, p3_])
    
    AD[:p1_, :p1_] = ADsub1
    AD[p1_:p2_, p1_:p2_] = ADsub2
    AD[p2_:p3_, p2_:p3_] = ADsub3
   
    D = np.zeros([p3_, p3])
    
    D[:p1_, :p1] = Dsub1
    D[p1_:p2_, p1:p2] = Dsub2
    D[p2_:p3_, p2:p3] = Dsub3
    
    U = np.zeros([p3, p3_])
    
    U[:p1, :p1_] = Usub1
    U[p1:p2, p1_:p2_] = Usub2
    U[p2:p3, p2_:p3_] = Usub3
    
    return A, AD, D, U



def getDenseMask(landmarks, h, w):
    # ... (function body from app.py)
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
    # ... (function body from app.py)
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
    # ... (function body from app.py)
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
    # ... (function body from app.py)
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
    hybrid.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device(device)))
    hybrid.eval()
    
    return hybrid


def pad_to_square(img):
    # ... (function body from app.py)
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
    # ... (function body from app.py)
    img, padding = pad_to_square(input_img)
    
    h, w = img.shape[:2]
    if h != 1024 or w != 1024:
        img = cv2.resize(img, (1024, 1024), interpolation = cv2.INTER_CUBIC)
        
    return img, (h, w, padding)


def removePreprocess(output, info):
    # ... (function body from app.py)
    h, w, padding = info
    
    if h != 1024 or w != 1024:
        output = output * h
    else:
        output = output * 1024
    
    padh, padw, auxh, auxw = padding
    
    output[:, 0] = output[:, 0] - padw//2
    output[:, 1] = output[:, 1] - padh//2
    
    return output   


def zip_files(files, zip_filename):
    # The zip function is modified to take a custom filename
    zip_path = os.path.join(OUTPUT_DIR, zip_filename)
    with ZipFile(zip_path, "w") as zipObj:
        for idx, file in enumerate(files):
            # Ensure the archive name is only the base filename, not the full path
            zipObj.write(file, arcname=os.path.basename(file)) 
    return zip_path


def segment(input_img_path):
    # The segment function is modified to create a unique sub-directory 
    # for each image's results.
    
    global hybrid, DEVICE # <-- FIX APPLIED HERE
    
    # 1. Check/Load Model
    if hybrid is None:
        print("Loading HybridGNet model...")
        hybrid = loadModel(DEVICE)
        print("Model loaded.")
    
    # 2. Setup output directory for this specific image
    base_name = os.path.splitext(os.path.basename(input_img_path))[0]
    local_output_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(local_output_dir, exist_ok=True)
    
    
    # 3. Preprocessing
    input_img = cv2.imread(input_img_path, 0) / 255.0
    if input_img is None:
        raise FileNotFoundError(f"Could not load image at {input_img_path}")
        
    original_shape = input_img.shape[:2]
    
    img, (h, w, padding) = preprocess(input_img)    
        
    data = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE).float()
    
    # 4. Inference
    with torch.no_grad():
        output = hybrid(data)[0].cpu().numpy().reshape(-1, 2)
        
    output = removePreprocess(output, (h, w, padding))
    output = output.astype('int')
    
    # 5. Save Outputs
    
    # Define paths within the specific image's output folder
    
    # Overlap Segmentation Image
    outseg = drawOnTop(input_img, output, original_shape) 
    seg_to_save = (outseg.copy() * 255).astype('uint8')
    overlap_path = os.path.join(local_output_dir, f"{base_name}_overlap_segmentation.png")
    cv2.imwrite(overlap_path , cv2.cvtColor(seg_to_save, cv2.COLOR_RGB2BGR))
    
    RL = output[0:44]
    LL = output[44:94]
    H = output[94:]
    
    # Landmark Text Files
    rl_path = os.path.join(local_output_dir, f"{base_name}_RL_landmarks.txt")
    ll_path = os.path.join(local_output_dir, f"{base_name}_LL_landmarks.txt")
    h_path = os.path.join(local_output_dir, f"{base_name}_H_landmarks.txt")
    
    np.savetxt(rl_path, RL, delimiter=" ", fmt="%d")
    np.savetxt(ll_path, LL, delimiter=" ", fmt="%d")
    np.savetxt(h_path, H, delimiter=" ", fmt="%d")
    
    # Binary Mask Images
    RL_mask, LL_mask, H_mask = getMasks(output, original_shape[0], original_shape[1])
    
    rl_mask_path = os.path.join(local_output_dir, f"{base_name}_RL_mask.png")
    ll_mask_path = os.path.join(local_output_dir, f"{base_name}_LL_mask.png")
    h_mask_path = os.path.join(local_output_dir, f"{base_name}_H_mask.png")
    
    cv2.imwrite(rl_mask_path, RL_mask)
    cv2.imwrite(ll_mask_path, LL_mask)
    cv2.imwrite(h_mask_path, H_mask)
    
    return local_output_dir


def batch_segmentation(input_dir):
    """
    Iterates through all supported image files in a directory and segments them.
    """
    
    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        return

    # Supported image extensions
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    image_paths = []
    for ext in extensions:
        # glob.glob is used to find all files matching the pattern (e.g., all *.jpg files)
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"No supported images found in the directory: {input_dir}")
        return

    print(f"Found {len(image_paths)} images to process.")

    # Process each image
    for path in image_paths:
        try:
            segment(path)
        except FileNotFoundError as e:
            print(f"Skipping {os.path.basename(path)}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {os.path.basename(path)}: {e}")
    
    print("\n--- Batch Segmentation Complete ---")



def load_hybrid():
    global hybrid_model
    if hybrid_model is None:
        print("Loading HybridGNet model...")
        hybrid_model = loadModel(DEVICE)  # your function from earlier
        print("HybridGNet loaded.")
    return hybrid_model

def load_classifier():
    global classifier_model
    if classifier_model is None:
        # NOTE: The load_model import is now at the top of the script
        classifier_model = load_model(MODEL_PATH, compile=False)
        print("Classifier loaded.")
    return classifier_model

# -------------------- HELPER FUNCTIONS --------------------
def calculate_mask_features(mask): 
    """
    Calculates the fractional area of heart and lungs from the mask.
    Mask convention: 1 for lung, 2 for heart.
    """
    mask = np.asarray(mask)
    heart_frac = np.mean(mask == 2) 
    lungs_frac = np.mean(mask == 1)
    return heart_frac, lungs_frac


def calculate_cdr_from_landmarks(rl_data, ll_data, h_data):
    """
    Calculates the Cardiopulmonary Ratio (CDR) from landmark data.
    """
    
    # --- 1. Helper function to parse data ---
    def parse_landmarks(data):
        """Parses the content string into a list of (Y, X) coordinates."""
        coords = []
        lines = data.strip().split('\n')
        for line in lines:
            if not line.strip(): continue
            try:
                # The segment function saves 'X Y', so read as 'X Y'
                x, y = map(int, line.strip().split())
                coords.append((y, x)) # Store as (Y, X) for easier indexing of rows/cols
            except ValueError: continue
        return coords

    # Load landmark data from the saved text files
    try:
        with open(rl_data, 'r') as f: rl_coords = parse_landmarks(f.read())
        with open(ll_data, 'r') as f: ll_coords = parse_landmarks(f.read())
        with open(h_data, 'r') as f: h_coords = parse_landmarks(f.read())
    except FileNotFoundError: return None

    if not rl_coords or not ll_coords or not h_coords: return None

    # Extract X (column/width) coordinates
    x_rl = [x for y, x in rl_coords]
    x_ll = [x for y, x in ll_coords]
    x_h = [x for y, x in h_coords]


    # --- 2. Calculate Thoracic Width (Max width of both lungs) ---
    x_thorax_max = max(x_ll) # X-coordinate of the rightmost point of the Left Lung
    x_thorax_min = min(x_rl) # X-coordinate of the leftmost point of the Right Lung
    thoracic_width = x_thorax_max - x_thorax_min

    # --- 3. Calculate Heart Width ---
    heart_width = max(x_h) - min(x_h)

    # --- 4. Calculate CDR ---
    if thoracic_width <= 0: return None
        
    cdr = heart_width / thoracic_width
    return cdr


def build_combined_mask_from_seg_folder(base, seg_folder, target_size):
    """
    Combines individual saved mask files into a single mask array.
    """
    
    # Read the saved mask files
    rl_mask_path = os.path.join(seg_folder, f"{base}_RL_mask.png")
    ll_mask_path = os.path.join(seg_folder, f"{base}_LL_mask.png")
    h_mask_path = os.path.join(seg_folder, f"{base}_H_mask.png")
    
    RL_mask = cv2.imread(rl_mask_path, 0)
    LL_mask = cv2.imread(ll_mask_path, 0)
    H_mask = cv2.imread(h_mask_path, 0)
    
    if RL_mask is None or LL_mask is None or H_mask is None:
        raise FileNotFoundError("One or more segmentation masks were not saved correctly.")
    
    # The convention from getDenseMask: 1 for lungs, 2 for heart
    combined_orig = np.zeros_like(RL_mask, dtype = np.uint8)
    
    # Lungs (value 1)
    combined_orig[RL_mask > 0] = 1
    combined_orig[LL_mask > 0] = 1
    
    # Heart (value 2). Heart overlaps lungs, so set it last.
    combined_orig[H_mask > 0] = 2
    
    # Resize to the target size for the classifier input
    combined_resized = cv2.resize(combined_orig.astype(np.float32), 
                                  target_size, 
                                  interpolation=cv2.INTER_NEAREST)
    
    return combined_orig, combined_resized.astype(np.uint8)


def compute_ctr(mask_folder, base):
    """
    Calculates the Cardiothoracic Ratio by reading landmark data from files.
    """
    rl_path = os.path.join(mask_folder, f"{base}_RL_landmarks.txt")
    ll_path = os.path.join(mask_folder, f"{base}_LL_landmarks.txt")
    h_path = os.path.join(mask_folder, f"{base}_H_landmarks.txt")
    
    cdr = calculate_cdr_from_landmarks(rl_path, ll_path, h_path)
    return cdr if cdr is not None else 0.5


# === GRADIO CLIENT FUNCTION (RAG) ===

def get_rag_answer_from_api(query: str):
    """Sends query to the external RAG server and returns the parsed response."""
    try:
        response = requests.get(RAG_SERVER_URL, params={"query": query}, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses
        
        data = response.json()
        if data.get("status") == "success":
            return data["data"]
        else:
            return {"answer": f"API Error: {data.get('message', 'Unknown error.')}", "sources": []}
            
    except requests.exceptions.RequestException as e:
        return {"answer": f"Connection Error: Could not reach RAG server at {RAG_SERVER_URL}. Is rag_server.py running in Terminal 1?", "sources": []}
def rag_cardiomegaly_qa(message, history):
    """
    Acts as the Gradio client, sending the query to the external RAG server,
    and returns the FULL response string instantly (NON-STREAMING) for stable rendering.
    """
    
    # 1. Get the answer from the external server
    result = get_rag_answer_from_api(message)
    
    answer = result.get('answer', 'Failed to retrieve an answer.')
    sources = result.get('sources', [])
    
    # 2. Format final response
    source_section = ""
    if sources:
        sources_list = "\n".join([f"* {name}" for name in sources])
        source_section = f"\n\n**Sources Consulted:**\n{sources_list}"

    final_answer = f"**Answer:** {answer}{source_section}"
    
    # 3. Print the full response to the console for verification (no streaming)
    print("\n\n--- RAG Response Complete (Client Output - Static) ---")
    print(final_answer)
    print("---------------------------------------------------\n")
    
    # RETURN the full string instantly. This guarantees it appears in Gradio.
    return final_answer

# -------------------- INITIALIZATION STUB --------------------
def initialize_rag():
    """Checks RAG server status and updates the Gradio status box."""
    RAG_CHECK_URL = "http://127.0.0.1:8000" # Base URL of the FastAPI server
    try:
        # Pings the /query endpoint with a dummy query
        response = requests.get(RAG_CHECK_URL + "/query?query=initialization_check", timeout=5)
        
        if response.status_code == 200 and response.json().get('status') == 'success':
             # The server is up and the RAG model is running
             return "RAG Server is ACTIVE and ready."
        else:
            # Server is up but initialization failed (e.g., ran out of memory)
            return f"WARNING: RAG Server reachable but reported an internal error (Status: {response.status_code})."

    except requests.exceptions.ConnectionError:
        return f"ERROR: RAG Server is offline. Please run 'python rag_server.py' in a separate terminal (Terminal 1) first."
    except Exception as e:
        return f"RAG Status Check Failed: {str(e)}"
    
def update_rag_context(context_text: str):
    """Send patient-specific context to the RAG server."""
    try:
        response = requests.post("http://127.0.0.1:8000/update_context", json={"context": context_text}, timeout=5)
        if response.status_code == 200:
            print("✅ Sent context to RAG server successfully.")
        else:
            print(f"⚠️ Failed to update RAG context: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"RAG context update error: {e}")    
def predict_cardiomegaly(image_path, heart_rate):
    """Full Cardiomegaly pipeline with segmentation, prediction, and RAG context update."""
    
    global hybrid, DEVICE
    if hybrid is None:
        print("Loading HybridGNet model for image prediction...")
        hybrid = loadModel(DEVICE)
        print("HybridGNet model loaded.")
    
    # --- 1. Segmentation ---
    seg_folder = segment(image_path)
    base = Path(image_path).stem

    # --- 2. Mask building + CDR ---
    combined_orig, combined_resized = build_combined_mask_from_seg_folder(
        base, seg_folder, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    ctr = compute_ctr(seg_folder, base)

    # --- 3. Mask-based features ---
    heart_frac, lungs_frac = calculate_mask_features(combined_resized)

    # --- 4. Prepare classifier inputs ---
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.0
    img_gray_resized = cv2.resize(img_gray, (IMG_WIDTH, IMG_HEIGHT))
    img_input = np.expand_dims(np.stack([img_gray_resized] * 3, axis=-1), axis=0)
    num_input = np.array([[ctr, heart_frac, lungs_frac]], dtype=np.float32)

    # --- 5. Predict with classifier ---
    classifier = load_classifier()
    prob = classifier.predict([img_input, num_input])[0][0]
    likelihood = prob * 100

    # --- 6. Interpret heart rate ---
    try:
        hr = float(heart_rate)
        if hr < 60:
            hr_status = f"Heart Rate: {hr:.0f} bpm — **Bradycardia (Too Low)**"
        elif hr > 100:
            hr_status = f"Heart Rate: {hr:.0f} bpm — **Tachycardia (Too High)**"
        else:
            hr_status = f"Heart Rate: {hr:.0f} bpm — **Normal Range**"
    except:
        hr = None
        hr_status = "Heart Rate: Not provided or invalid."

    # --- 7. Diagnostic logic ---
    if ctr > 0.50 and likelihood > 50.0:
        diagnosis = (
            f"**HIGH RISK**: Susceptible to Cardiomegaly ({likelihood:.2f}%). "
            f"CDR = {ctr:.4f} (above 0.50)."
        )
    elif ctr > 0.50:
        diagnosis = (
            f"**BORDERLINE**: Elevated CDR ({ctr:.4f}) but low model confidence ({likelihood:.2f}%). "
            "Clinical review advised."
        )
    elif likelihood > 75.0:
        diagnosis = (
            f"**HIGH RISK**: Model confidence is high ({likelihood:.2f}%) despite normal/low CDR ({ctr:.4f})."
        )
    else:
        diagnosis = f"**LOW RISK**: Likely no Cardiomegaly ({likelihood:.2f}%). CDR = {ctr:.4f}."

    # --- 8. Prepare and send clean RAG context ---
    # 🩺 Important: use the exact key format expected by rag_server regex
    rag_context = (
        f"cdr: {ctr:.4f}\n"
        f"likelihood: {likelihood:.2f}%\n"
        f"heart rate: {hr if hr is not None else 'N/A'}"
    )
    update_rag_context(rag_context)

    # --- 9. Load output images for display ---
    overlap_path = os.path.join(seg_folder, f"{base}_overlap_segmentation.png")
    rl_mask_path = os.path.join(seg_folder, f"{base}_RL_mask.png")
    ll_mask_path = os.path.join(seg_folder, f"{base}_LL_mask.png")
    h_mask_path = os.path.join(seg_folder, f"{base}_H_mask.png")

    overlap_img = cv2.cvtColor(cv2.imread(overlap_path), cv2.COLOR_BGR2RGB)
    rl_mask_img = cv2.imread(rl_mask_path, 0)
    ll_mask_img = cv2.imread(ll_mask_path, 0)
    h_mask_img = cv2.imread(h_mask_path, 0)

    # --- 10. Output JSON ---
    output_data = {
        "Cardiothoracic Ratio (CDR)": f"{ctr:.4f}",
        "Prediction Likelihood": f"{likelihood:.2f}%",
        "Final Assessment": diagnosis,
        "Heart Rate Status": hr_status
    }

    return overlap_img, rl_mask_img, ll_mask_img, h_mask_img, output_data
# -------------------- GRADIO INTERFACE --------------------
requests.post("http://127.0.0.1:8000/clear_context")

# Define the components for the X-ray analysis tab
xray_analysis_inputs = [
    gr.File(file_types=[".png", ".jpg", ".jpeg"], label="Upload Chest X-ray (PA View)"),
    gr.Number(label="Heart Rate (bpm)", value=75, info="Enter the patient’s heart rate for reference")
]

xray_analysis_outputs = [
    gr.Image(type="numpy", label="Overlap Segmentation (Heart: Cyan, Lungs: Magenta)"),
    gr.Image(type="numpy", label="Right Lung Mask"),
    gr.Image(type="numpy", label="Left Lung Mask"),
    gr.Image(type="numpy", label="Heart Mask"),
    gr.JSON(label="Cardiomegaly Prediction & Metrics"),
]


# Create the Gradio Interface with Tabs
with gr.Blocks(title="HybridGNet Cardiomegaly AI") as iface:
    gr.Markdown("# HybridGNet Cardiomegaly Prediction and Research Assistant 🩺")
    gr.Markdown("This tool provides two core functions: **1. X-ray Analysis** for image-based cardiomegaly prediction and **2. Research QA** for text-based information.")
    
    with gr.Tab("1. X-ray Analysis (Image Prediction)"):
        gr.Interface(
            fn=predict_cardiomegaly,
            inputs=xray_analysis_inputs,
            outputs=xray_analysis_outputs,
            title="Chest X-ray Cardiomegaly Prediction",
            description="Upload a PA chest X-ray. The model segments the heart and lungs, calculates the Cardiothoracic Ratio (CDR), and predicts cardiomegaly likelihood.",
            allow_flagging="never"
        )
        
    with gr.Tab("2. Research QA (RAG Assistant)"):
        # Display the knowledge base status first
        kb_status = gr.Textbox(label="RAG Server Status", interactive=False)
        gr.Markdown("---")
        gr.Markdown("## Q&A Chat Interface")

        # Chat Interface for Q&A
        gr.ChatInterface(
            fn=rag_cardiomegaly_qa,
            chatbot=gr.Chatbot(height=500),
            textbox=gr.Textbox(placeholder="Ask about Cardiomegaly, CDR, or a topic found in the research papers...", container=False, scale=7),
            title="Cardiomegaly Research Q&A",
            description=f"This chat uses an **external server** running in a separate terminal. Ensure 'rag_server.py' is running on {RAG_SERVER_URL}.",
            theme="soft",
            submit_btn="Send",
            # clear_btn is intentionally REMOVED 
        )
        
    # Initialization step for RAG when the interface starts (checks if server is running)
    iface.load(initialize_rag, None, kb_status)

# Use the Blocks object as the main interface to launch
if __name__ == "__main__":
    
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    iface.launch(share=True)