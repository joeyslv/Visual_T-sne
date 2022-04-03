import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

embeddings = {
    "t-SNE embeedding": TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        n_iter=500,
        n_iter_without_progress=100,
        n_jobs=2,
        random_state=0,
    ),
}

def dataload(path_org):
    W = True
    if os.path.isdir(path_org):
        files=os.listdir(path_org)
        num_org = len(files)
        org_y = np.array([0]*num_org)
        org_img = np.array([])
        i = 1
        print('\n')
        for file_name in files:
            file_path=os.path.join(path_org,file_name)
            img = Image.open(file_path).convert('L')
            img = img.resize((128,128),Image.BILINEAR)
            img = np.asarray([np.array(img)])
            if W == True:
                org_img = img
                W =False
            else:
                org_img = np.vstack((org_img,img))
            print(f"\r读取{path_org}下图片：%.2f%%" %(float(i/num_org*100)),end=' ')
            i=i+1
        print('total:', num_org)
    return org_img.reshape(num_org,-1),org_y
        
def plot_embedding(X, title, ax,org_y,color,color1,color2):
    X = MinMaxScaler().fit_transform(X)
    for oy in org_y:
        scatter = ax.scatter(
                *X[org_y == oy].T,
                marker="o",
                s=20,
                alpha=0.3,
                zorder=2,
                c=color,
                edgecolors = 'face'
            ) 
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='org',
                        markerfacecolor=color1, markersize=8),
                    Line2D([0], [0], marker='o', color='w', label='syn',
                    markerfacecolor=color2, markersize=8)]

    ax.legend(handles=legend_elements, loc='lower right')
    ax.set_title(title)
    # ax.axis("off")
    
def show(org_x,org_y,syn_x,syn_y):
    projections, timing = {}, {}
    for name, transformer in embeddings.items():
        data = org_x
        print(f"\nComputing {name}...")
        start_time = time()
        projections[name] = transformer.fit_transform(data, org_y)
        fake_X = transformer.fit_transform(syn_x, syn_y)
        timing[name] = time() - start_time

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    for name in timing:
        title = f"{name} (time {timing[name]:.3f}s)"
        plot_embedding(projections[name], title,axs,org_y,'#9467bd','#9467bd','#bcbd22')
        plot_embedding(fake_X, title,axs,syn_y,'#bcbd22','#9467bd','#bcbd22')

def main():

    
    #path_org: the real pic path
    #path_syn: the generating pic path
    path_org = 'E:/Object-Detection/data_radar/temp/jpg'
    path_syn = 'E:/Object-Detection/data_radar/temp/jpg'

    org_x , org_y = dataload(path_org)
    syn_x , syn_y = dataload(path_syn)
    show(org_x,org_y,syn_x,syn_y)
    plt.savefig("t-sne.png",dpi=600)
    plt.show()
    

if __name__ == '__main__':
    main()
