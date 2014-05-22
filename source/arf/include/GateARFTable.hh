/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#ifndef GateARFTable_h
#define GateARFTable_h

#include"globals.hh"
#include <vector>
#include <map>

#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TBranch.h"

class TH2I;
class TH2D;
class TH1D;

class GateARFTable
{

private:

G4String m_name;       // the name of the ARF Table
G4int m_Index;         // the index of the AERF Table because many are defined for one simulation
G4double*  m_theTable; // contains the probability for each cos(theta)_k, tan(phi)_k' as a single linearized index
G4double*  m_theDRFTable;  // contains the ARF table before normalization
G4int m_drfdimx;
G4double fDist_src2img;
G4int  iAvgPixNum;
G4double m_drfbinsize;
G4double m_lowX;
G4double m_lowY;
G4int m_drfdimy;
G4int m_NbOfcosTheta ; // the number of discretized values of cos(theta)
G4int m_NbOftanPhi;    // the number of discretized values of tan(phi)
G4int m_TotalNb;       // the product of m_NbOfcosTheta by m_NbOftanPhi
G4double m_Elow;      // the left end energy of the energy window
G4double m_Ehigh;      // the right end energy of the energy window
G4int m_isprimary;
G4double m_ElowOut; // window energy specified by the user
G4double m_EhighOut;// window energy specified by the user

G4double* m_theta;
G4double* cosTheta; 
G4double* cosTheta_i ;
G4double* tanPhi;
G4double* tanPhi_i;
G4double* Phi;
G4double dStep1,dStep2,dStep3,dStep4,dTanPhiStep1; // setpes parameters for the theta,phi grids
G4double dBase1,dBase2,dBase3,dBase4;
G4double dErgReso, dResoAtErg;

G4double dConstantTerm,dLinearTerm;

G4double m_TotSimuPhotons;  // total number of simulated photons for this incident energy window


long unsigned int m_counter; //  the number of binned photons

  //long unsigned int m_TotPhotons;
  //long unsigned int m_rejected;
int iphicounts;

public:
GateARFTable(G4String);
~GateARFTable();
G4String GetName() { return m_name;};
void GetARFAsBinaryBuffer(G4double*&);
void FillTableFromBuffer(G4double*&);
void SetNSimuPhotons(G4double N)
{m_TotSimuPhotons = N;G4cout<<" TOTAL number of photons   " <<(long unsigned int)(m_TotSimuPhotons)<<G4endl; };
G4int GetPrimary() { return m_isprimary;};
void SetPrimary() { m_isprimary = 1;}
void SetNoPrimary() { m_isprimary = 0;}
G4double GetEWlow() { return m_ElowOut; };
G4double GetEWhigh() { return m_EhighOut; };
void SetName(G4String aName){ m_name = aName; };
G4int GetIndex() { return m_Index; };
void SetIndex( G4int aIndex ) { m_Index = aIndex; };
G4int GetNbofTheta() { return m_NbOfcosTheta; };
G4int GetNbofPhi() { return m_NbOftanPhi; };
G4int GetTotalNb() { return m_TotalNb; };
G4double GetElow() { return m_Elow; };
void SetElow( G4double aD ) { m_Elow = aD; };
G4double GetEhigh() { return m_Ehigh; };
void SetEhigh( G4double aD ) { m_Ehigh = aD; };
void Describe();
void Initialize(G4double,G4double);
G4int GetIndexes( G4double,G4double, G4int& ,G4int& );
void NormalizeTable();
void StoreYZToRootFile(G4double,G4double,G4int,G4int);
G4double RetrieveProbability(G4double , G4double );
void SetEnergyReso(G4double);
void SetERef(G4double);
inline G4double GetEnergyReso(){return dErgReso;};
inline G4double GetERef(){return dResoAtErg;};
void FillDRFTable( G4double , G4double  , G4double  );
void convertDRF2ARF();
G4double computeARFfromDRF( G4double , G4double , G4double );
void SetDistanceFromSourceToDetector( G4double aD ){fDist_src2img = aD; };
};
#endif

#endif // G4ANALYSIS_USE_ROOT
