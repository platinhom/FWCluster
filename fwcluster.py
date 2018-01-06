#! /usr/bin/env python
# -*- coding: utf-8 -*

"""
Generate Framework and the framework based clustering.

Dependent: RDKit

Author : Zhixiong Zhao
Update : 2017.1.5

Version 0.1 : 

"""

#%%
DEBUG_MODE = True

import os, sys, rdkit, StringIO
import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import *
from rdkit.Chem import Draw
from rdkit.Chem import FragmentCatalog, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from rdkit.Chem.Draw import MolDrawing, DrawingOptions
DrawingOptions.includeAtomNumbers=False

def mol2mpl(mol, size=60):
    '''Simple func to depict molecule'''
    Draw.MolToMPL(mol, (size,size)).show()


def GetFramework(mol, retainDblBond=True, replace2SingleBnd=False, replace2Carbon=False, \
                smallestRing=False, returnIdx=False, returnSmiles=False):
    '''Get the Framework of given molecule.
    
    mol: Given input molecule
    retainDblBond: if True, retain double bond to framework. 
                   Not working when replace2SingleBnd is True.
    replace2SingleBnd: if True, Replace bonds to single bond. 
                       If set to 2, also replace for aromatic.  
    replace2Carbon: if True, replace non-C atom to Carbon. 
                    If set to 2, also replace in ring.  
    smallestRing: if True, Contract the ring to smallest ring.
                  It can't work on Aromatic bond. Better use with replace2SingleBnd(True).

    returnIdx: if True, return idx of framework atoms instead of new molecule.
    returnSmiles: if True, return SMILES instead of new molecule.
    
    Return: 
        Default to return new molecule for framework.
        If returnIdx=True, return ascending list of framework atom idx.
        If returnSmiles=True, return SMILES instead of new molecule.
    '''

    #replace2SingleBnd=2
    #retainDblBond=False
    #replace2Carbon=True
    #smallestRing=True

    #suppl = Chem.SmilesMolSupplier(fname)
    #mol = MolFromSmilesSupplItem(suppl,5)
    
    #mol = Chem.MolFromSmiles('CC(C)C(C(=O)NC(=NC)C1CCCN1C(=O)C(C)NC(=O)C(C)NC(=O)CCC(=O)OC)NC2=CN=C(C=C2)[N+](=O)[O-]')
    #mol = Chem.MolFromSmiles('CCCCC(CCC)CCC(=C(C)C)CCCCC')
    #mol2mpl(mol,300)

    ## frameatoms: save framework atom idx in set()
    frameatoms=set()
    sssr=Chem.GetSymmSSSR(mol)
    ringcounts=len(sssr)
    if ringcounts>=1:
        ####### Contain ring Molecule
        ## Find the Murcko Scaffold to update frameatoms     
        scaffold=MurckoScaffold.GetScaffoldForMol(mol)

        ####### Don't retain outside double bond #######
        ### Avoid outside double bond in Murcko Scaffold if Not to retain DblBond or to replace all to singleBnd
        if not retainDblBond or replace2SingleBnd:
            ### May be better in other method
            outside=set()
            map(outside.update, scaffold.GetSubstructMatches(Chem.MolFromSmarts('[$([D1]=*)]')))
            for idx in outside:
                for bond in scaffold.GetAtomWithIdx(idx).GetBonds():
                    if bond.GetBondTypeAsDouble()>1.0 : 
                        bond.SetBondType(rdkit.Chem.rdchem.BondType.SINGLE)
            scaffold=MurckoScaffold.GetScaffoldForMol(scaffold)
        ####### Don't retain outside double bond #######

        scaffoldmatch= mol.GetSubstructMatches(scaffold)
        map(frameatoms.update, scaffoldmatch) 

        ## Get Atom Distance Matrix for analysis
        dm = Chem.GetDistanceMatrix(mol)
        while True:
            newscaffold=set()
            framelist=sorted(frameatoms)
            dmscaffold=dm[:,framelist]
            dist=dict()
            for i in range(dm.shape[0]):
                if i not in frameatoms:
                    dmsmin=dmscaffold[i].min()
                    ## If Atom distance to framework >= 4, Maybe As framework
                    if int(dmsmin) >= 4:
                        #### TODO: Whether judge Carbon?
                        ## Nearest framework atom Idx
                        nearest = framelist[dmscaffold[i].argmin()]
                        if dist.has_key(nearest):
                            dist[nearest].append((i,int(dmsmin)))
                        else:
                            dist.setdefault(nearest,[(i,int(dmsmin))])

            ## Only keep the longest path to given framework atom
            for key,value in dist.iteritems():
                maxdist=np.array(value)[:,1].max()
                maxatoms=[]
                for item in value:
                    if maxdist == item[1]:
                        maxatoms.append(item[0])                       
                for maxatom in maxatoms:
                    newscaffold.update(Chem.GetShortestPath(mol,maxatom,key))

            ## If find new framework, update and next loop
            if len(newscaffold)>0:
                frameatoms.update(newscaffold)
            else:
                break
            
    else:
        #####  No ring Molecule
        maxpathpair=set()
        dm = Chem.GetDistanceMatrix(mol)
        diameter = int(np.max(dm))
        diameterfloat = np.max(dm)
        for i in range(dm.shape[0]):
            if diameterfloat in dm[i]:
                maxpathpair.add(tuple(sorted([i, list(dm[i]).index(diameterfloat)])))
        for maxpair in maxpathpair:
            frameatoms.update(Chem.GetShortestPath(mol,maxpair[0],maxpair[1]))

    ####### Retain X=X bond in framework based on bond analysis
    ## Can't retain such as Scaffold>C=N-CC information
    if retainDblBond and not replace2SingleBnd:
        doublebondatoms=set()
        for atom in mol.GetAtoms():
            idx=atom.GetIdx()
            if idx not in frameatoms:
                nears=atom.GetNeighbors()
                for near in nears:
                    idxnear=near.GetIdx()
                    if idxnear in frameatoms:
                        bond=mol.GetBondBetweenAtoms(idx,idxnear)
                        if bond.GetBondTypeAsDouble()>1.0:
                            doublebondatoms.add(idx)
        frameatoms.update(doublebondatoms)
    #print frameatoms

    ## To depict atomic Idx  
    if returnIdx:
        return sorted(frameatoms)
    else:
        ## Get framework molecule
        newmol=Chem.EditableMol(mol)
        ## Delete should from last idx
        for i in range(mol.GetNumAtoms()-1,-1,-1):
            if i not in frameatoms:
                newmol.RemoveAtom(i)
        ## Avoid Error after editable molecule
        newmol=Chem.MolFromSmiles(Chem.MolToSmiles(newmol.GetMol()))
 
        ######### Replace to Single Bond
        if replace2SingleBnd:
            for bond in newmol.GetBonds():# and (replace2SingleBnd==2 or not bond.GetIsAromatic())
                if bond.GetBondTypeAsDouble()>1.0: 
                    bond.SetBondType(rdkit.Chem.rdchem.BondType.SINGLE)
            newmol.UpdatePropertyCache()
            Chem.SanitizeMol(newmol)

        ######### Replace to Carbon
        if replace2Carbon:
            query=Chem.MolFromSmarts('[!R;!#6]')
            replace=Chem.MolFromSmiles('C')
            newmol=Chem.ReplaceSubstructs(mol=newmol,query=query, replacement=replace, replaceAll=True)[0]
            if replace2Carbon==2:
                for atom in newmol.GetAtoms():
                    if atom.IsInRing() and atom.GetAtomicNum() != 6:
                        atom.SetAtomicNum(6)
                        atom.UpdatePropertyCache()
            newmol.UpdatePropertyCache()
            Chem.SanitizeMol(newmol)

        ########## Replace Ring to Smallest Ring ########
        if smallestRing:
            control=True
            while control:
                control=False
                for atom in newmol.GetAtoms():
                    if atom.IsInRing() and not atom.GetIsAromatic() and atom.GetExplicitValence()>=2 and len(atom.GetNeighbors())==2:
                        idx1=atom.GetNeighbors()[0].GetIdx()
                        idx2=atom.GetNeighbors()[1].GetIdx()
                        if not newmol.GetBondBetweenAtoms(idx1,idx2):
                            newmol2=Chem.EditableMol(newmol)
                            newmol2.AddBond(idx1,idx2,order=rdkit.Chem.rdchem.BondType.SINGLE)
                            newmol2.RemoveAtom(atom.GetIdx())
                            newmol=Chem.MolFromSmiles(Chem.MolToSmiles(newmol2.GetMol()))
                            control=True
                            break

        #mol2mpl(newmol,300)
        if returnSmiles:
            return Chem.MolToSmiles(newmol)
        else:
            return newmol


def ClusterFps(fps,cutoff=0.2, metric='Tanimoto'):
    '''Clustering Structure based on given Fingerprints.
    fps: Fingerprint Input for clustering.
    cutoff: Cutoff for Butina Clustering.
    metric: Available similarity metrics include:
        Tanimoto, Dice, Cosine, Sokal, Russel, Kulczynski, McConnaughey, and Tversky.
    '''
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    metricsAvailableBulk={'tanimoto':DataStructs.BulkTanimotoSimilarity,"dice":DataStructs.BulkDiceSimilarity,
    "cosine": DataStructs.BulkCosineSimilarity, "sokal": DataStructs.BulkSokalSimilarity, "russel": DataStructs.BulkRusselSimilarity, 
    "rogotGoldberg": DataStructs.BulkRogotGoldbergSimilarity, "allbit": DataStructs.BulkAllBitSimilarity, 
    "kulczynski": DataStructs.BulkKulczynskiSimilarity, "mcconnaughey": DataStructs.BulkMcConnaugheySimilarity,
    "asymmetric": DataStructs.BulkAsymmetricSimilarity, "braunblanquet": DataStructs.BulkBraunBlanquetSimilarity}
    
    if metric.lower() not in metricsAvailableBulk:
        print "The given metric is unknown!"
        metric='Tanimoto'
    simMetricsBulk=metricsAvailableBulk[metric.lower()]

    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1,nfps):
        sims = simMetricsBulk(fps[i],fps[:i])
        dists.extend([1-x for x in sims])

    # now cluster the data:
    cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return cs

def ClusterOnFingerprint(filename, mols=None, fingerprint=0, cutoff=0.8, metric='Tanimoto', outMatrix=False):
    '''Clustering Structure based on Fingerprints in RDKit

    filename: Smile format file saving molecules. If set to None, use given "mols"
    mols: Input molecules. No use if set up "filename"
    cutoff: Cutoff using for Butina Clustering
    fingerprint: Fingerprint to use:
        0 or else:  RDKit Topological Fingerprint
        1: MACCS Fingerprint
        2: Atom Pair Fingerprint (AP)
        3: Topological Torsion Fingerprint (TT)
        4: Morgan Fingerprint similar to ECFP4 Fingerprint
        5: Morgan Fingerprint similar to FCFP4 Fingerprint
    metric: Available similarity metrics include: 
            Tanimoto, Dice, Cosine, Sokal, Russel, Kulczynski, McConnaughey, and Tversky.
    outMatrix: Change output to a similarity matrix
    Return: Default output "clusters, clusterOut":
        clusters: Clusters containing molecule number.
        clusterOut: Molecular Cluster Number in List.
    '''

    from rdkit import DataStructs
    from rdkit.Chem.Draw import SimilarityMaps
    from rdkit.Chem.Fingerprints import FingerprintMols
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem.AtomPairs import Pairs, Torsions

    if filename:
        suppl = Chem.SmilesMolSupplier(filename)
        mols=[]
        for mol in suppl:
            mols.append(mol)
    molnums=len(mols)

    ### Calculate Molecular Fingerprint
    ## MACCS Fingerprint
    if fingerprint==1:
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
    ## Atom Pair Fingerprint (AP)
    elif fingerprint == 2:
        fps = [Pairs.GetAtomPairFingerprint(mol) for mol in mols]
    ## Topological Torsion Fingerprint (TT)
    elif fingerprint == 3:
        fps = [Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol) for mol in mols]
    ## Morgan Fingerprint similar to ECFP4 Fingerprint
    elif fingerprint == 4:
        fps = [AllChem.GetMorganFingerprint(mol,2) for mol in mols]
    ## Morgan Fingerprint similar to FCFP4 Fingerprint
    elif fingerprint == 5:
        fps = [AllChem.GetMorganFingerprint(mol,2,useFeatures=True) for mol in mols]
    ## RDKit Topological Fingerprint
    else: #fingerprint==0:
        fps = [FingerprintMols.FingerprintMol(mol) for mol in mols]

    if outMatrix:
        ### Output the Fingerprint similarity Matrix
        metricsAvailable={'tanimoto':DataStructs.TanimotoSimilarity,"dice":DataStructs.DiceSimilarity,
        "cosine": DataStructs.CosineSimilarity, "sokal": DataStructs.SokalSimilarity, "russel": DataStructs.RusselSimilarity, 
        "rogotGoldberg": DataStructs.RogotGoldbergSimilarity, "allbit": DataStructs.AllBitSimilarity, 
        "kulczynski": DataStructs.KulczynskiSimilarity, "mcconnaughey": DataStructs.McConnaugheySimilarity,
        "asymmetric": DataStructs.AsymmetricSimilarity, "braunblanquet": DataStructs.BraunBlanquetSimilarity}
        
        if metric.lower() not in metricsAvailable:
            print "The given metric is unknown!"
            metric='Tanimoto'

        simMetrics=metricsAvailable[metric.lower()]

        ### Calculate Fingerprint similarity Matrix
        simdm=[[0.0]*molnums]*molnums
        for i in range(molnums):
            simdm[i,i]=1.0
            for j in range(i+1,molnums):
                simdm[i,j]=DataStructs.FingerprintSimilarity(fps[i],fps[j],metric=simMetrics)
                simdm[j,i]=DataStructs.FingerprintSimilarity(fps[j],fps[i],metric=simMetrics)

        for i in range(molnums):
            print
            for j in range(molnums):
                print '%3.2f' % simdm[i,j],
        return simdm

    else:
        clusters=ClusterFps(fps, cutoff=1-cutoff, metric='Tanimoto')
        clusterID=0
        clusterOut=[0]*len(mols)
        for cluster in clusters:
            clusterID+=1
            for idx in cluster:
                clusterOut[idx]=clusterID
            ## To depict cluster molecule
            if False:
                if len(cluster)>1:
                    print "Cluster: "
                    for idx in cluster:
                        mol2mpl(mols[idx])
        return clusters, clusterOut      

def FrameworkClustering(filename, mode=1, retainDblBond=True, 
    replace2SingleBnd=False, replace2Carbon=False, smallestRing=False,
    fingerprint=0, cutoff=0.8, metric='Tanimoto', outMatrix=False):

    '''Clustering based on Molecular Framework

    filename: Smile format file saving molecules. 
    outMatrix: Change output to a similarity matrix
    Return: Default output "clusters, clusterOut":
        clusters: Clusters containing molecule number.
        clusterOut: Molecular Cluster Number in List.

    mode: Mode for Framework generation. 
        0: No framework.
        1 or else: Default, Cut sidechain, Saving outside double bonds.
            In this mode, use retainDblBond, replace2SingleBnd, replace2Carbon, smallestRing
        2: Similar to 1 but Don't save outside double bonds.
        3: Similar to 2 but replace non-aromatic bond to single bond.
        4: Similar to 3 but replace all to carbon.
        5: Similar to 4 but contract the smallest ring.
    retainDblBond: if True, retain double bond to framework. 
                   Not working when replace2SingleBnd is True.
    replace2SingleBnd: if True, Replace bonds to single bond. 
                       If set to 2, also replace for aromatic.  
    replace2Carbon: if True, replace non-C atom to Carbon. 
                    If set to 2, also replace in ring.  
    smallestRing: if True, Contract the ring to smallest ring.
                  It can't work on Aromatic bond. Better use with replace2SingleBnd(True).

    cutoff: Cutoff using for Butina Clustering
    fingerprint: Fingerprint to use:
        0 or else:  RDKit Topological Fingerprint
        1: MACCS Fingerprint
        2: Atom Pair Fingerprint (AP)
        3: Topological Torsion Fingerprint (TT)
        4: Morgan Fingerprint similar to ECFP4 Fingerprint
        5: Morgan Fingerprint similar to FCFP4 Fingerprint
    metric: Available similarity metrics include: 
            Tanimoto, Dice, Cosine, Sokal, Russel, Kulczynski, McConnaughey, and Tversky.
    '''

    suppl = Chem.SmilesMolSupplier(filename)
    mols=[]
    for mol in suppl:
        mols.append(mol)
    molnums=len(mols)

    ## Framework generation. Preset mode or given args (mode=1)
    if mode == 0:
        frameworks = mols
    elif mode == 2:
        frameworks = [ GetFramework(mol, retainDblBond=False) for mol in mols ]
    elif mode == 3:
        frameworks = [ GetFramework(mol, retainDblBond=False, replace2SingleBnd=True) for mol in mols ]
    elif mode == 4:
        frameworks = [ GetFramework(mol, retainDblBond=False, replace2SingleBnd=True, replace2Carbon=True) for mol in mols ]
    elif mode == 5:
        frameworks = [ GetFramework(mol, retainDblBond=False, replace2SingleBnd=True, replace2Carbon=True, smallestRing=True) for mol in mols ]
    else:
        frameworks = [ GetFramework(mol, retainDblBond=retainDblBond, replace2SingleBnd=replace2SingleBnd, replace2Carbon=replace2Carbon, smallestRing=smallestRing) for mol in mols ]
        
    if outMatrix:
        simdm = ClusterOnFingerprint(filename=None, mols=frameworks, fingerprint=fingerprint, \
        cutoff=cutoff, metric=metric, outMatrix=outMatrix)
        return simdm

    else: 
        clusters, clusterOut = ClusterOnFingerprint(filename=None, mols=frameworks, fingerprint=fingerprint, \
        cutoff=cutoff, metric=metric, outMatrix=outMatrix)

        if False:
            for cluster in clusters:
                if len(cluster)>1:
                    for idx in cluster:
                        mol2mpl(mols[idx])

        return clusters, clusterOut
