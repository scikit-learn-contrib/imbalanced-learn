import numpy as np
from sklearn.decomposition import PCA,KernelPCA

#LICIC: Less Important Components for Imbalanced multi-class Classification
#Vincenzo Dentamaro, Donato Impedovo, Giuseppe Pirlo
#vincenzo@gatech.edu, donato.impedovo@uniba.it, giuseppe.pirlo@uniba.it
#MDPI Information October 2018

class LICIC:

    def __init__(self,noise = 0.99, kernel = 'linear', ratio = 1.0,k = 2, components_ratio = 0.2,gamma= None,degree = None,components = None):
        self.noise = noise
        self.kernel = kernel
        self.ratio = ratio
        self.k = k
        self.components_ratio = components_ratio
        self.gamma = gamma
        self.components = components
        self.ratio = ratio
        self.degree = degree

    def fit_sample(self,x,y):
        return self._minority_resample(x,y,self.noise,self.kernel,self.ratio,self.k,self.components_ratio,self.gamma,self.degree,self.components_ratio)


    def _kpca_minority(self,data,noise = 0.99,kernel = 'linear',gamma = None, degree = 3,components = None):
            import  copy
            data_copy = copy.deepcopy(data)
            if components != None:
                if kernel == 'linear':
                    kpca = PCA()
                else:
                    kpca = KernelPCA(kernel=kernel,gamma=gamma,degree=degree,fit_inverse_transform=True)
                X_train_kpca = kpca.fit_transform(data)
                if kernel == 'linear':
                    explVar = kpca.explained_variance_ratio_
                else:
                    explVar = kpca.lambdas_/kpca.lambdas_.sum()
                v = 0
                comp = 0
                for k in range(len(explVar)):
                    v += explVar[k]
                    if v < noise:
                        comp = k
                comp += 1
            else:
                comp = components
            if kernel == 'linear':
                kpca2 = PCA(n_components=comp)
            else:
                kpca2 = KernelPCA(kernel=kernel,gamma=gamma,degree=degree,n_components=comp,fit_inverse_transform=True)

            transformed = kpca2.fit_transform(data_copy)

            return (transformed, kpca2)

    def _resampling(self,diff, data, k,components_ratio = 0.3):
        from sklearn.neighbors import NearestNeighbors
        nb = NearestNeighbors(n_neighbors = k,metric='euclidean').fit(data)


        new_data = []
        start_last_components = len(data[0]) - int(len(data[0])*components_ratio)
        components_to_play_with = len(data[0])-start_last_components
        for j in range(diff):
            idx = np.random.randint(0,len(data))
            sample = data[idx]
            components = {}
            sample_ = sample.reshape(1,len(sample))
            distances, indices = nb.kneighbors(sample_)

            # #new
            # distances = []
            # for j in range(k):
            #     distances.append(j)
            # temp = []
            # for i in range(k):
            #     idx = np.random.randint(0,len(data))
            #     temp.append(idx)
            # indices = []
            # indices.append(temp)
            # #end new


            for i in range(len(distances)):
                idx = indices[0][i]

                sample2 = data[idx]
                picked_components = []
                for h in range(int(components_to_play_with/len(distances))):
                    go = True
                    while go:
                        cpix = np.random.randint(start_last_components,start_last_components+components_to_play_with)
                        if cpix not in picked_components:
                            go = False
                            picked_components.append(cpix)

                    sample[cpix] = sample2[cpix]
                    #diff_comp = sample[cpix] - sample2[cpix]

                    # if cpix in components:
                    #      components[cpix].append(diff_comp)
                    # else:
                    #      components[cpix] = []
                    #      components[cpix].append(diff_comp)
                    #sample[cpix] = sample2[cpix]
            # for key in components:
            #     gap = np.random.uniform(0,1)
            #     sample[key] = sample[key]+gap*np.array(components[key]).mean()

            new_data.append(sample)

        new_data = np.array(new_data)
        if len(new_data) > 0:
            return np.vstack((data,new_data))
        else:
            return data



    def _minority_resample(self,trainingset,ytrain,noise = 0.99, kernel = 'linear', ratio = 1.0,k = 2, components_ratio = 0.2,gamma= None,degree = None,components = None):
        if ratio < 1.0:
            return None
        classes = np.unique(ytrain)
        training = {}
        max_length = 0
        (transf_data,kpca) = self._kpca_minority(trainingset,noise=noise, kernel=kernel,components = components,gamma=gamma,degree=degree)
        training_ret = []
        y = []
        for i in range(len(classes)):
            clas = classes[i]
            indices = np.where(ytrain == clas)
            training[clas] = {}
            training[clas]['data'] = transf_data[indices]
            if len(training[clas]['data']) > max_length:
                max_length = len(training[clas]['data'])


        max_length = int(max_length * ratio)
        for i in range(len(classes)):
            clas = classes[i]
            class_object = training[clas]

            data = class_object['data']

            diff = max_length-len(data)
            data = self._resampling(diff,data, k=k,components_ratio=components_ratio)
            data = kpca.inverse_transform(data)
            for t in range(len(data)):
                training_ret.append(data[t])
                y.append(clas)


        training_ret = np.array(training_ret)
        y = np.array(y)
        return training_ret,y
