#ifndef GateRTVPhantom_HH
#define GateRTVPhantom_HH 1

#include <stdio.h>          
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "globals.hh"

#include "GateRTPhantom.hh"




class GateRTVPhantomMessenger;

class GateRTVPhantom : public GateRTPhantom
{


private:
GateRTVPhantomMessenger* m_messenger;
G4int NbOfFrames;
G4String base_FN; // base file name of frames
G4String current_FN; // current Frame file name
G4String header_FN; // the header file name for reading the matrix dimensions
G4int cK; // current index for phantoms file
G4double m_TPF; // time per frame
G4int set_ActAsAtt;
G4int set_AttAsAct;
public:
GateRTVPhantom();
G4int GetNbOfFrames();
void   SetNbOfFrames( G4int aNb );
void   SetBaseFileName( G4String aFN );
void   SetHeaderFileName( G4String aFN );
void   SetActAsAtt(){ set_ActAsAtt = 1;}
void   SetAttAsAct(){ set_AttAsAct = 1;}

void SetTPF( G4double aTPF);
G4double GetTPF();

void Compute(G4double aTime);

};

#endif

