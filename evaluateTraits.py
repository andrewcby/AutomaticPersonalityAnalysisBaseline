
# coding: utf-8

import numpy as np
import csv


def evaluateTraits(p, gt):
    if (len(p) == len(gt)):
        for i in range(len(p)):
            if (len(p[i]) != 5) or (len(gt[i]) != 5):
                print "Inputs must be a list of 5 values within the range [0,1]. Traits could not be evaluated."
                return
            for j in range(len(p[i])):
                if p[i][j] < 0 or p[i][j] > 1 or gt[i][j] < 0 or gt[i][j] > 1:
                    print "Inputs must be values in the range [0,1]. Traits could not be evaluated."
                    return
    
    errors = np.abs(p-gt)
    meanAccs = 1-np.mean(errors, axis=0)
    
    print "\nAverage accuracy of "+str(np.mean(meanAccs))+": "
    
    # These scores are reported.
    print "Accuracy predicting Extraversion: "+str(meanAccs[0])
    print "Accuracy predicting Agreeableness: "+str(meanAccs[1])
    print "Accuracy predicting Conscientiousness: "+str(meanAccs[2])
    print "Accuracy predicting Neuroticism: "+str(meanAccs[3])
    print "Accuracy predicting Openness to Experience: "+str(meanAccs[4])
    print "\n"
    
    # m_p = np.mean(p*100, axis=0)
    # m_gt = np.mean(p*100, axis=0)
    # traitsp = getNominalTraits(m_p)
    # traitsgt = getNominalTraits(m_gt)
    
    # print "Ground Truth: "
    # print str(m_gt[0])+"% Extraversion: Tends to be "+traitsgt[0]
    # print str(m_gt[1])+"% Agreeableness: Tends to be "+traitsgt[1]
    # print str(m_gt[2])+"% Conscientiousness: Tends to be "+traitsgt[2]
    # print str(m_gt[3])+"% Neuroticism: Tends to be "+traitsgt[3]
    # print str(m_gt[4])+"% Openness to Experience: Tends to be "+traitsgt[4]
    
    # print "\nPredicted: "
    # print str(m_p[0])+"% Extraversion: Tends to be "+traitsp[0]
    # print str(m_p[1])+"% Agreeableness: Tends to be "+traitsp[1]
    # print str(m_p[2])+"% Conscientiousness: Tends to be "+traitsp[2]
    # print str(m_p[3])+"% Neuroticism: Tends to be "+traitsp[3]
    # print str(m_p[4])+"% Openness to Experience: Tends to be "+traitsp[4]
    
    return meanAccs


def getNominalTraits(traits):
    
    traitsname = []
    
    if traits[0] < .5: traitsname.append("Introvert")
    elif traits[0] > .5: traitsname.append("Extravert")
    else: traitsname.append("Ambivert")
    
    if traits[1] < .5: traitsname.append("Challenger")
    elif traits[1] > .5: traitsname.append("Adapter")
    else: traitsname.append("Negotiator")
        
    if traits[2] < .5: traitsname.append("Flexible")
    elif traits[2] > .5: traitsname.append("Focused")
    else: traitsname.append("Balanced")
    
    if traits[3] < .5: traitsname.append("Resilient")
    elif traits[3] > .5: traitsname.append("Reactive")
    else: traitsname.append("Responsive")
    
    if traits[4] < .5: traitsname.append("Preserver")
    elif traits[4] > .5: traitsname.append("Explorer")
    else: traitsname.append("Moderate")
    
    return traitsname


# Read the files with the scores
def readScores(filepath):
    results = []
    with open(filepath, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            results.append([float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5])])
    csvfile.close()
    return results    


def main():
    # Example of testing the evaluation of traits with randomly generated predictions and ground truth.
    # p = np.random.rand(5,1);
    # gt = np.random.rand(5,1);
    
    p = np.asarray(readScores("./predictions.csv"))
    gt = np.asarray(readScores("./validation_gt_full_name.csv"))
    
    results = evaluateTraits(p, gt)

if __name__ == "__main__":
    main()