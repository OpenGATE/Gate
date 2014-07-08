/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GateARFTableMgr_h
#define GateARFTableMgr_h

#include "globals.hh"
#include<map>
#include "G4ThreeVector.hh"



class GateARFSD;
class GateARFTable;
class GateARFTableMgrMessenger;


class GateARFTableMgr
{

private:

std::map<G4int,GateARFTable*> m_theList;
GateARFTableMgrMessenger* m_messenger;
G4int m_currentIndex;
G4int m_verboseLevel;
G4String m_name;
G4double Energy[40];
GateARFSD* m_theARFSD;
G4String m_filename;

G4double m_EThreshHold;
G4double m_EUpHold;
G4double m_ERef;
G4double m_EnReso;
G4double m_distance;

G4int SaveARFTables;
G4int LoadARFTables;
G4String theFN;
G4int m_nbins;
public:
 GateARFTableMgr( G4String, GateARFSD* );
 ~GateARFTableMgr();
void SaveARFToBinaryFile();
void SetBinaryFile(G4String theName ){SaveARFTables = 1; theFN = theName;}; 
void LoadARFFromBinaryFile(G4String);
void SetNBins(G4int);
G4int GetNBins(){return m_nbins;};
void SetEThreshHold( G4double aET) { m_EThreshHold = aET;};
void SetEUpHold( G4double aEU) { m_EUpHold = aEU;};
void SetEReso( G4double );
void SetERef(G4double);
G4int GetCurrentIndex() { return m_currentIndex; };
void AddaTable(GateARFTable* aTable);
void SetVerboseLevel(G4int aL){ m_verboseLevel = aL ;};
void AddaTable( G4String );
void ComputeARFTables(G4int);
void ComputeARFTablesFromEW(G4String);
void ListTables();
G4String GetName() { return m_name;};
void SetName( G4String aName ){m_name = aName; };
GateARFSD* GetARFSD() { return m_theARFSD; };
G4int InitializeTables();
void FillDRFTable( G4int,G4double,G4double,G4double );
void SetNSimuPhotons(G4double*);
void convertDRF2ARF();
void CloseARFTablesRootFile();
G4double ScanTables( G4double, G4double, G4double);
void SetDistanceFromSourceToDetector( G4double aD ){ m_distance = aD;};
};

#endif





