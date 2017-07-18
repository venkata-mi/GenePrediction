"""
Created on Mon Jun 26 2017.

@author: Venkata Sai Sriram Pillutla

"""

from bs4 import BeautifulSoup
import urllib
import pandas as pd

#FTP URL for PATRIC
PatricURL = 'ftp://ftp.patricbrc.org/patric2/genomes/'

def getGenomeSequence(genomeId):
    """
    This method fetches the genome sequence based on genomeid from PATRIC

    Parameter: genomeId
    """
    
    r = urllib.urlopen(PatricURL+genomeId+'/'+genomeId+'.fna').read()
    soup = BeautifulSoup(r)
    #print type(soup)

    genomeSequence = soup.prettify().split('| '+genomeId+']')[1]
    return genomeSequence.strip('\n')


def getFeaturesForGenome(genomeId, CDS_ONLY):
    """
    This method gets the features for a particular genomeId frfom PATRIC

    Parameters

    genomeId: UniqueId for the genome
    CDS_ONLY: retrieve only CDS features
    """
    data_table = pd.read_table(PatricURL
                               +genomeId+'/'+genomeId+'.PATRIC.features.tab')

    
    print data_table.shape

    if CDS_ONLY:
        return data_table[(data_table.feature_type == 'CDS')]
        
    else:
        return data_table

 



