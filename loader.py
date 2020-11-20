import torch
import csv
import json
import random
import os.path
from data import *
import numpy as np
import pandas as pd

"""
How the Class Loader works is as follows:
[0] pathfinder returns the path to the feature files of a specified track, 
[1] hitCollector(df) and nonhitCollector(df) gather data from the files of lowlevel and highlevel features and finally make a csv file.
    if the csv file already exists it returns the content of the file either as a numpy array or a datafram depending on the input "df".
[2] we need to concat hit and nonhit songs after we collect them because for onehot encoding and quantizing we need to see all 
the values in the dataset and collectAll(df) does this.
[3] onehotEncoder(df) gets the names of the columns that needs to be encoded and encodeAll(df) applies it to all the dataset
[4] The following functions returns the number of the columns for highlevel features, encoded features and so on.
[5] quantizer(bins, col_values) gets the number of bins return a quantized version with the name of the bins.
    getLowBins(bins) applies this function on lowlevel features and return a onehot encoded of the lowlevel features based on their bins
    getYearBins(bins) does the same thing for year values
[6] allHighFeatures() and allLowFeatures(bins) concat different types of high/low level features and return them as a dataframe
[7] dataLoader(year_bins, low_bins) concats ids,labels,highlevel features and lowlevel features together and finally seperate the hit and 
    nonhit songs and return them as numpy arrays along with col_names
[8] The first column of the arrays contains the ids, the second col contains the label
    the following (yearbins + 59) contains the highlevel features and the rest are all lowlevel features

About Class OneMillion:
It calls the dataLoader on a Loader object and gets the numpy array of hit_songs and nonhit_songs features along with 
their labels and ids. you need to determine the number of bins for year and lowlevel features quantizing.
the first column of the numpy array contains the ids
the second column contains the labels
the rest of the columns contains all features in the format of 0/1
"""

class OneMillion():
    def __init__(self,is_test = False):
        loader = Loader('../../msd_bb_matches.csv', '../../msd_bb_non_matches.csv','../../msd_audio_features','.')
        hit_songs, non_hit_songs, col_names,ids,labels = loader.dataLoader(year_bins=10, low_bins=4)
        #ids, labels, feautures
 
        #please take care of the code from here
        random_list_hits = np.random.randint(0,hit_songs.shape[0],int(hit_songs.shape[0]*0.3))
        random_list_non_hits = np.random.randint(0,non_hit_songs.shape[0],int(non_hit_songs.shape[0]*0.3))
        

        if is_test == True:
            self.hit_songs = hit_songs[random_list_hits,:]
            self.non_hit_songs = non_hit_songs[random_list_non_hits,:]
            hit_labels = labels[:len(hit_songs)]
            non_hit_labels = labels[len(hit_songs):]
            labels = np.concatenate((hit_labels[random_list_hits],non_hit_labels[random_list_non_hits]),axis = 0)
            
        else:
            self.hit_songs = np.delete(hit_songs,random_list_hits,axis = 0) 
            self.non_hit_songs = np.delete(non_hit_songs,random_list_non_hits,axis = 0)
            hit_labels = labels[:len(hit_songs)]
            non_hit_labels = labels[len(hit_songs):]
            labels = np.concatenate((np.delete(hit_labels, random_list_hits,axis = 0),np.delete(non_hit_labels,random_list_non_hits,axis = 0)),axis = 0)
            
        self.data = np.concatenate((self.hit_songs,self.non_hit_songs), axis = 0)

        self.ids = ids
        self.labels =  np.eye(2)[labels.astype(np.int64)]
        self.col_names = col_names
#        print(len(self.hit_songs))
#        print(len(self.labels))
#        print(len(self.non_hit_songs))
         
 
    def __getitem__(self,index):
        return torch.from_numpy(self.data[index]), torch.from_numpy(self.labels[index]),self.ids[index]


    def __len__(self):
        return self.data.shape[0]


class Loader:

    def __init__(self, hitfile, nonhitfile, datapath, resultpath):
        self.hitfile = hitfile
        self.nonhitfile = nonhitfile
        self.datapath = datapath
        self.resultpath = resultpath 
    
    """returns the path of low level and high level features for a given track id"""
    def pathFinder(self, track_id):
        directory_letter = track_id[2].lower()
        lowPath = ''
        highPath = ''
        
        lowPath = self.datapath + '/features_tracks_' + directory_letter + '/' + track_id + '.lowlevel.json'
        highPath = self.datapath + '/features_tracks_' + directory_letter + '/' + track_id + '.mp3.highlevel.json'
        return [lowPath , highPath]

    """returns the values of artist, title, and year for a given index in a given dataframe"""
    def additionalFeatures(self, df, index):
        artist = df.at[index , 'artist']
        title = df.at[index , 'title']
        year =  df.at[index , 'year']
        return [artist, title, year]      

    """creates a csv file of hit songs' features or return the dataframe if the file already exists"""
    def hitCollector(self, df=False):
        if os.path.exists('hit.csv'):
            if(df):
                return pd.read_csv('hit.csv', index_col=0)
            else:
                return pd.read_csv('hit.csv', index_col=0).to_numpy()
        matches = pd.read_csv(self.hitfile)
        matches = matches.sort_values(by = ['msd_id'])
        matches = matches.reset_index(drop=True)
        hitData = pd.DataFrame()
        msd_ids = list(matches['msd_id'])
        for tid in msd_ids:
            lowPath , highPath = self.pathFinder(tid) 
            if(os.path.exists(lowPath) and os.path.exists(highPath)):
                addfeatures = self.additionalFeatures(matches, msd_ids.index(tid))
                track = Data(lowPath, highPath)
                trackData = np.array( [tid, 1] + addfeatures + track.concatValues())
                trackLabel = ['id', 'hit', 'artist', 'title', 'year'] + track.concatFeatures()
                trackdf = pd.DataFrame([trackData], columns = trackLabel)
                hitData = hitData.append(trackdf, ignore_index=True)
        hitData.to_csv(self.resultpath + '/hit.csv')
        if(df):
            return hitData
        return (hitData.reset_index()).to_numpy()
        
    """creates a csv file of nonhit songs' features or return the dataframe if the file already exists"""    
    def nonhitCollector(self, numbers ,df=False):
        if os.path.exists('non_hits.csv'):
            if(df):
                return pd.read_csv('non_hits.csv', index_col=0)
            else:
                return pd.read_csv('non_hits.csv', index_col=0).to_numpy()
        nonmatches = pd.read_csv(self.nonhitfile)
        nonmatches = nonmatches.sort_values(by = ['msd_id'])
        nonmatches = nonmatches.reset_index(drop=True)
        nonhitData = pd.DataFrame()
        msd_ids = list(nonmatches['msd_id'])
        count = 0
        randomTracks = random.sample(range(len(msd_ids)), numbers)
        for i in randomTracks:
            tid = msd_ids[i]
            lowPath , highPath = self.pathFinder(tid) 
            if(os.path.exists(lowPath) and os.path.exists(highPath)):
                addfeatures = self.additionalFeatures(nonmatches, i)
                track = Data(lowPath, highPath)
                trackData = np.array( [tid, 0] + addfeatures + track.concatValues())
                trackLabel = ['id', 'hit', 'artist', 'title', 'year'] + track.concatFeatures()
                trackdf = pd.DataFrame([trackData], columns = trackLabel)
                nonhitData = nonhitData.append(trackdf, ignore_index=True)
        nonhitData.to_csv(self.resultpath + '/non_hits.csv')
        if(df):
            return nonhitData
        return (nonhitData.reset_index()).to_numpy()

    """merges nonhit and hit songs dataframe"""        
    def collectAll(self, df=False):
        hitData = self.hitCollector(True)
        nonhitData = self.nonhitCollector(hitData.shape[0],True)
        allData = hitData.append(nonhitData, ignore_index=True)
        #allData.to_csv(self.resultpath + '/all.csv')
        if(df):
            return allData
        return (allData.reset_index()).to_numpy()

    """gets a dataframe of songs and encode the nan features to one hot"""    
    def oneHotEncoder(self, df):
        nan_features = ["chords_key", "chords_scale", "key_key", "key_scale", "genre_dortmund", "genre_electronic", "genre_rosamerica", "genre_tzanetakis", "ismir04_rhythm", "moods_mirex"]
        encoded_df = pd.get_dummies(df, columns=nan_features)
        return encoded_df
    
    """ get the csv file of hit songs and rewrite the file with a onehot-encoded version for nan features"""
    def encodeAll(self, df=False):
        allData = self.collectAll(True)
        allData = self.oneHotEncoder(allData)
        if(df):
            return allData.reset_index()
        return (allData.reset_index()).to_numpy()  
    
    
    """returns the list of columns including numerical highlevel features in the csv"""
    def highColumns(self):
        return list(range(6, 18))
    
    """returns the list of columns including onehot-encoded highlevel features in the csv"""
    def highOnehotColumns(self):
        return list(range(482, 529))
    
    """returns the list of all columns including highlevel features, both numerical and an onehot-encoded"""
    def allHighColumns(self):
        return self.highColumns() + self.highOnehotColumns()
    
    """returns the list of columns including numerical lowlevel features"""
    def lowColumns(self):
        return list(range(18,454))
    
    """returns the list of columns including onehot-encoded lowlevel features"""
    def lowOnehotColumns(self):
        return list(range(454, 482))
    
    """returns the list of all columns including lowlevel features, both numerical and an onehot-encoded"""
    def allLowColumns(self):
        return self.lowColumns() + self.lowOnehotColumns()

    """quantize the float values into bins"""
    def quantize(self, bins, col_values):
        max_val = np.amax(col_values)
        min_val = np.amin(col_values)
        interval = (max_val - min_val)/bins
        quantized = []
        for i in range(len(col_values)):
            if (col_values[i] == max_val):
                quantized.append("bin"+str(bins))
            elif (col_values[i] == min_val):
                quantized.append("bin"+str(1))
            else:
                for j in range(bins): 
                    if (col_values[i] >= (min_val + j * interval) and col_values[i] < (min_val + (j+1) * interval)):
                        quantized.append("bin"+str(j+1))
        
        return np.array(quantized)
    
    """get highlevel 0/1 matrix"""
    def allHighFeatures(self):
        labels = self.encodeAll(True).columns[self.allHighColumns()]
        all_np = self.encodeAll(False)
        return pd.DataFrame(all_np[:,self.allHighColumns()], columns=labels)

    """get lowlevel 0/1 matrix"""
    def getLowBins(self, bins):
        lowLabels = self.encodeAll(True).columns[self.lowColumns()]
        allData = self.encodeAll(False)
        quantized_low = []
        for c in self.lowColumns():
            quantized_col = self.quantize(bins, allData[:,c])
            quantized_low.append(quantized_col)
        return pd.get_dummies(pd.DataFrame(np.transpose(np.array(quantized_low)), columns=lowLabels))
    
    """concat onehot low features with quantized low features"""
    def allLowFeatures(self, bins):
        low_bins = self.getLowBins(bins)
        onehotLabels = self.encodeAll(True).columns[self.lowOnehotColumns()]
        low_onehot = pd.DataFrame(self.encodeAll(False)[:,self.lowOnehotColumns()], columns=onehotLabels)
        return pd.concat([low_onehot, low_bins], axis=1)

    """get quantized years"""
    def getYearBins(self, bins):
        allData = self.encodeAll(False)
        quantized_year = pd.get_dummies(pd.DataFrame(self.quantize(bins, allData[:,5]), columns=["year"]))
        return quantized_year

    """load all data and seperate the hit songs from nonhit songs"""
    def dataLoader(self, year_bins, low_bins):
        hit_no = len(self.hitCollector(False))
        ids = self.encodeAll(True)['id']
        labels = self.encodeAll(True)['hit']
        all_songs = pd.concat([self.getYearBins(year_bins), self.allHighFeatures(), self.allLowFeatures(low_bins)], axis=1)
        hit_songs = all_songs.iloc[:hit_no, :]
        nonhit_songs = all_songs.iloc[hit_no:, :]
        cols = all_songs.columns
        return hit_songs.to_numpy(dtype = np.float64), nonhit_songs.to_numpy(dtype = np.float64), cols.to_numpy(), ids.to_numpy(), labels.to_numpy(dtype = np.float32)
    

#myloader = Loader('../../msd_bb_matches.csv', '../../msd_bb_non_matches.csv', '../../msd_audio_features', '.')
#myloader.collectAll()
