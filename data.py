import json


class Data:

    """
        initiate the object with parsed json files of the features
    """
    def __init__(self, lowPath, highPath):
        self.lowPath = lowPath
        self.highPath = highPath
        with open(lowPath) as json_file:
            self.lowLevel = json.load(json_file)
        with open(highPath) as json_file:
            self.highLevel = json.load(json_file)

    ##### HIGH LEVEL
    """
    getting a full list of high level features names
    """
    def highList(self):
        return list(self.highLevel["highlevel"].keys())

    """
    getting all the possible values for a high level feature
    """
    def highAll(self, feature):
        return list(self.highLevel["highlevel"][feature]["all"].keys())

    """
    getting the value of a high level feature
    """
    def highValue(self, feature):
        return self.highLevel["highlevel"][feature]['value']
      
    """
    getting the list of binary features
    """
    def highBinaryFeatures(self):
        binFeatures  = []
        for f in self.highList():
            if(len(self.highAll(f)) == 2):
                binFeatures.append(f)
        return binFeatures
            
    """ 
    getting the binary value (0, 1) for a binary feature
    """   
    def highBinaryValue(self, feature):
        val = self.highValue(feature)
        return (int)(not(self.highAll(feature).index(val)))

    """
    getting the list of non binary high level features
    """
    def highNonbinFeatures(self):
        nonbinFeatures = []
        for f in self.highList():
            if f not in self.highBinaryFeatures():
                nonbinFeatures.append(f)
        return nonbinFeatures

    def highFeaturesConcat(self):  
        return self.highBinaryFeatures() + self.highNonbinFeatures()

    def highValuesConcat(self):
        highValues = []
        for f in self.highFeaturesConcat():
            if(f in self.highBinaryFeatures()):
                highValues.append(self.highBinaryValue(f))
            else:
                highValues.append(self.highValue(f))
        return highValues
    
    ##### LOW LEVEL

    def lowList(self):
        return list(self.lowLevel["lowlevel"].keys())

    def lowListCleaned(self):
        lowFeatures = self.lowList()
        lowFeatures.remove("barkbands")
        lowFeatures.remove("erbbands")
        lowFeatures.remove("gfcc")
        lowFeatures.remove("melbands")
        lowFeatures.remove("mfcc")
        lowFeatures.remove("spectral_contrast_coeffs")
        lowFeatures.remove("spectral_contrast_valleys")
        return lowFeatures
    
    def lowSingleValue(self):
        return ["average_loudness", "dynamic_complexity"]

    def lowMultiValue(self):
        multiValueFeatures = self.lowListCleaned()
        for f in self.lowSingleValue():
            multiValueFeatures.remove(f)
        return multiValueFeatures

    def lowDmean(self, feature):
        if(feature in self.lowMultiValue()):
            return self.lowLevel["lowlevel"][feature]["dmean"]

    def lowDmean2(self, feature):
        if(feature in self.lowMultiValue()):
            return self.lowLevel["lowlevel"][feature]["dmean2"] 

    def lowDmean(self, feature):
        if(feature in self.lowMultiValue()):
            return self.lowLevel["lowlevel"][feature]["dmean"] 

    def lowDvar(self, feature):
        if(feature in self.lowMultiValue()):
            return self.lowLevel["lowlevel"][feature]["dvar"]

    def lowDvar2(self, feature):
        if(feature in self.lowMultiValue()):
            return self.lowLevel["lowlevel"][feature]["dvar2"]

    def lowMax(self, feature):
        if(feature in self.lowMultiValue()):
            return self.lowLevel["lowlevel"][feature]["max"]

    def lowMean(self, feature):
        if(feature in self.lowMultiValue()):
            return self.lowLevel["lowlevel"][feature]["mean"]

    def lowMedian(self, feature):
        if(feature in self.lowMultiValue()):
            return self.lowLevel["lowlevel"][feature]["median"]

    def lowMin(self, feature):
        if(feature in self.lowMultiValue()):
            return self.lowLevel["lowlevel"][feature]["min"]
     

    def lowVar(self, feature):
        if(feature in self.lowMultiValue()):
            return self.lowLevel["lowlevel"][feature]["var"]

    def lowMainFeatures(self):
        return  self.lowSingleValue() + self.lowMultiValue()

    def lowFeaturesConcat(self):  
        features = self.lowSingleValue()
        for f in self.lowMultiValue():
            for k in list(self.lowLevel['lowlevel'][f].keys()):
                features.append(f + "." + k)
        return features
        
    def lowValuesConcat(self):
        lowValues = []
        for f in self.lowMainFeatures():
            if(f in self.lowSingleValue()):
                lowValues.append(self.lowLevel['lowlevel'][f])
            else:
                lowValues.append(self.lowDmean(f))
                lowValues.append(self.lowDmean2(f))
                lowValues.append(self.lowDvar(f))
                lowValues.append(self.lowDvar2(f))
                lowValues.append(self.lowMax(f))
                lowValues.append(self.lowMean(f))
                lowValues.append(self.lowMedian(f))
                lowValues.append(self.lowMin(f))
                lowValues.append(self.lowVar(f))
        return lowValues

    ##### RHYTHM

    def rhythmList(self):
        return list(self.lowLevel["rhythm"].keys())

    def rhythmListCleaned(self):
        rhythmFeatures = self.rhythmList()
        rhythmFeatures.remove("beats_loudness_band_ratio")
        rhythmFeatures.remove("beats_position")
        return rhythmFeatures

    def rhythmSingleValue(self):
        return ["beats_count", "bpm", "danceability", "onset_rate"]

    def rhythmMultiValue(self):
        multiValueFeatures = self.rhythmListCleaned()
        for f in self.rhythmSingleValue():
            multiValueFeatures.remove(f)
        return multiValueFeatures

    def rhythmDmean(self, feature):
        if(feature in self.rhythmMultiValue()):
            return self.lowLevel["rhythm"][feature]["dmean"]
     

    def rhythmDmean2(self, feature):
        if(feature in self.rhythmMultiValue()):
            return self.lowLevel["rhythm"][feature]["dmean2"]
     

    def rhythmDmean(self, feature):
        if(feature in self.rhythmMultiValue()):
            return self.lowLevel["rhythm"][feature]["dmean"]
     

    def rhythmDvar(self, feature):
        if(feature in self.rhythmMultiValue()):
            return self.lowLevel["rhythm"][feature]["dvar"]
     

    def rhythmDvar2(self, feature):
        if(feature in self.rhythmMultiValue()):
            return self.lowLevel["rhythm"][feature]["dvar2"]
     

    def rhythmMax(self, feature):
        if(feature in self.rhythmMultiValue()):
            return self.lowLevel["rhythm"][feature]["max"]
     

    def rhythmMean(self, feature):
        if(feature in self.rhythmMultiValue()):
            return self.lowLevel["rhythm"][feature]["mean"]
     

    def rhythmMedian(self, feature):
        if(feature in self.rhythmMultiValue()):
            return self.lowLevel["rhythm"][feature]["median"]
     

    def rhythmMin(self, feature):
        if(feature in self.rhythmMultiValue()):
            return self.lowLevel["rhythm"][feature]["min"]
     

    def rhythmVar(self, feature):
        if(feature in self.rhythmMultiValue()):
            return self.lowLevel["rhythm"][feature]["var"]
    
    def rhythmMainFeatures(self):
        return  self.rhythmSingleValue() + self.rhythmMultiValue()

    def rhythmFeaturesConcat(self):  
        features = self.rhythmSingleValue()
        for f in self.rhythmMultiValue():
            for k in list(self.lowLevel['rhythm'][f].keys()):
                features.append(f + "." + k)
        return features

    def rhythmValuesConcat(self):
        rhythmValues = []
        for f in self.rhythmMainFeatures():
            if(f in self.rhythmSingleValue()):
                rhythmValues.append(self.lowLevel['rhythm'][f])
            else:
                rhythmValues.append(self.rhythmDmean(f))
                rhythmValues.append(self.rhythmDmean2(f))
                rhythmValues.append(self.rhythmDvar(f))
                rhythmValues.append(self.rhythmDvar2(f))
                rhythmValues.append(self.rhythmMax(f))
                rhythmValues.append(self.rhythmMean(f))
                rhythmValues.append(self.rhythmMedian(f))
                rhythmValues.append(self.rhythmMin(f))
                rhythmValues.append(self.rhythmVar(f))
        return rhythmValues
     

    ##### TONAL 

    def tonalList(self):
        return list(self.lowLevel["tonal"].keys())

    def tonalListCleaned(self):
        tonalFeatures = self.tonalList()
        tonalFeatures.remove("hpcp")
        tonalFeatures.remove("chords_histogram")
        tonalFeatures.remove("thpcp")
        return tonalFeatures

    def tonalMultiValue(self):
        return ["chords_strength", "hpcp_entropy"]

    def tonalSingleValue(self):
        singleValueFeatures = self.tonalListCleaned()
        for f in self.tonalMultiValue():
            singleValueFeatures.remove(f)
        return singleValueFeatures

    def tonalDmean(self, feature):
        if(feature in self.tonalMultiValue()):
            return self.lowLevel["tonal"][feature]["dmean"] 

    def tonalDmean2(self, feature):
        if(feature in self.tonalMultiValue()):
            return self.lowLevel["tonal"][feature]["dmean2"]

    def tonalDmean(self, feature):
        if(feature in self.tonalMultiValue()):
            return self.lowLevel["tonal"][feature]["dmean"]

    def tonalDvar(self, feature):
        if(feature in self.tonalMultiValue()):
            return self.lowLevel["tonal"][feature]["dvar"]

    def tonalDvar2(self, feature):
        if(feature in self.tonalMultiValue()):
            return self.lowLevel["tonal"][feature]["dvar2"] 

    def tonalMax(self, feature):
        if(feature in self.tonalMultiValue()):
            return self.lowLevel["tonal"][feature]["max"]
     
    def tonalMean(self, feature):
        if(feature in self.tonalMultiValue()):
            return self.lowLevel["tonal"][feature]["mean"]

    def tonalMedian(self, feature):
        if(feature in self.tonalMultiValue()):
            return self.lowLevel["tonal"][feature]["median"]

    def tonalMin(self, feature):
        if(feature in self.tonalMultiValue()):
            return self.lowLevel["tonal"][feature]["min"]

    def tonalVar(self, feature):
        if(feature in self.tonalMultiValue()):
            return self.lowLevel["tonal"][feature]["var"]

    def tonalMainFeatures(self):
        return  self.tonalSingleValue() + self.tonalMultiValue()

    def tonalFeaturesConcat(self):
        features =  self.tonalSingleValue()
        for f in self.tonalMultiValue():
            for k in list(self.lowLevel['tonal'][f].keys()):
                features.append(f + "." + k)
        return features
        

    def tonalValuesConcat(self):
        tonalValues = []
        for f in self.tonalMainFeatures():
            if(f in self.tonalSingleValue()):
                tonalValues.append(self.lowLevel['tonal'][f])
            else:
                tonalValues.append(self.tonalDmean(f))
                tonalValues.append(self.tonalDmean2(f))
                tonalValues.append(self.tonalDvar(f))
                tonalValues.append(self.tonalDvar2(f))
                tonalValues.append(self.tonalMax(f))
                tonalValues.append(self.tonalMean(f))
                tonalValues.append(self.tonalMedian(f))
                tonalValues.append(self.tonalMin(f))
                tonalValues.append(self.tonalVar(f))
        return tonalValues

   
    ##### CONCAT FEATURES
    def concatFeatures(self):
        labels = self.highFeaturesConcat() + self.lowFeaturesConcat() + self.rhythmFeaturesConcat() + self.tonalFeaturesConcat()
        return labels

    def concatValues(self):
        values = self.highValuesConcat() + self.lowValuesConcat() + self.rhythmValuesConcat() +  self.tonalValuesConcat()
        return values
        

#mydata = Data("/mnt/d/TRAZZZP128F424AC28.lowlevel.json" , "/mnt/d/TRAZZZP128F424AC28.mp3.highlevel.json")



