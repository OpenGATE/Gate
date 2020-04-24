/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/* ----------------------------------------------------------------------------- * 
 *                                                                         *
 *  GateSingleParticleSourceMessenger remplace  GateGeneralParticleSourceM-*
 *  essenger                                                               *
 *                                                                         *
 * ----------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------- *
 *                                                                         *
 *  \file GateGeneralParticleSourceMessenger.hh                            *
 *                                                                         *
 *  Revision 1.5 2014/08/1 Yann PERROT and Simon NICOLAS                   *
 *  Add command: "setUserSpectrumCmd"                                      *
 *  This command allows to load files describing energies and probabilities*
 *  from a user spectrum.                                                     *                                                                                                        
 *                                                                         *
 *  $Log: GateGeneralParticleSourceMessenger.hh,v $                        *
 *  Revision 1.4  2008/03    dbenoit and fcassol                           * 
 *  Update the GateSingleParticleSourceMessenger                           *
 *                                                                         * 
 *  Revision 1.3  2002/10/11 14:52:38  lsimon                              *
 *  Bug fix: the destruction of the UI directory is not left to the        *
 *  base-class GateMessenger, to avoid a double-attempt of destruction     *
 *                                                                         *
 *  Revision 1.2  2002/08/11 15:33:24  dstrul                              *
 *  Cosmetic cleanup: standardized file comments for cleaner doxygen output*
 *                                                                         *
 *  \brief Class GateGeneralParticleSourceMessenger                        *
 *  \brief By Giovanni.Santin@cern.ch                                      *
 *  \brief $Id: GateGeneralParticleSourceMessenger.hh,v 1.3 2002/10/11     * 
 *  14:52:38 lsimon Exp $                                                  *
 *                                                                         *
 *  \brief Class GateSingleParticleSourceMessenger                         *
 *  \brief By benoit@cppm.in2p3.fr and cassol@cppm.in2p3.fr                *
 *                                                                         *
 * ----------------------------------------------------------------------------- */
 
/* ----------------------------------------------------------------------------- *
 *                                                                         *
 *  A large part of the code below is a direct copy of the content of the  *
 *  G4Tokenizer class from Geant4.4.0. It is thus the intellectual property*
 *  of the Geant4 collaboration and is submitted to their disclaimer       *
 *                                                                         *
 * ----------------------------------------------------------------------------- */
 
/* ----------------------------------------------------------------------------- *
 *                                                                         *
 *  Class Description :                                                    *
 *                                                                         *
 *  The function of the GateSingleParticleSourceMessenger is to allow the  *
 *  user to enter commands either in interactive command line mode or      *
 *  through macros to control the G4SingleParticleSource. the              *
 *  GateGeneralParticleSourceMessenger class is based on                   *
 *  G4ParticleGunMessenger.                                                *
 *                                                                         *
 * ----------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------- *
 *                                                                         *
 *  MEMBER FUNCTIONS :                                                     *
 *                                                                         *
 *  GateSingleParticleSourceMessenger(GateSingleParticleSource* fPtclGun)  *
 *    Constructor:  Sets up commands.                                      *
 *                                                                         *
 *  ~GateGeneralParticleSourceMessenger()                                  *
 *    Destructor:  Deletes commands.                                       *
 *                                                                         *
 *  void SetNewValue(G4UIcommand *command, G4String newValues)             *
 *    Uses the appropriate methods in the G4SingleParticleSource to carry  *
 *    out the user commands.                                               *
 *                                                                         *
 *  G4String GetCurrentValue(G4UIcommand *command)                         *
 *    Allows the user to retrieve the current values of parameters.        *
 *    Not implemented yet.                                                 *
 *                                                                         *
 * ----------------------------------------------------------------------------- */

#ifndef GATESINGLEPARTICLESOURCEMESSENGER_H
#define GATESINGLEPARTICLESOURCEMESSENGER_H 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateMessenger.hh"

class G4ParticleTable;
class G4UIcommand;
class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithABool;
class G4UIcmdWithoutParameter;
class GateVSource;

//-------------------------------------------------------------------------------------------------
class GateSingleParticleSourceMessenger : public GateMessenger
{

 public:
  
  GateSingleParticleSourceMessenger( GateVSource* fPtclGun ) ;
  ~GateSingleParticleSourceMessenger() ;
    
  void SetNewValue( G4UIcommand* command, G4String newValues ) ;
      /*Identifies the command which has been invoked by the user, extracts 
        the parameters associated with that command (held in newValues), and 
	uses these values with the appropriate member function of 
	G4SingleParticleSource.*/
  
  G4String GetCurrentValue( G4UIcommand* command ) ;
  inline G4bool GetIonShooting() const                 { return fShootIon; }

 private:

  void IonCommand( G4String newValues ) ;

 private:

  //G4ParticleTable* particleTable ;
  G4String histtype ;
  G4int    fNArbEHistPoints;
  G4bool   fArbInterModeSet;
  GateVSource* fParticleGun ;
     
 private: //commands

   // positional commands
  G4UIdirectory              *positionDirectory;
  G4UIcmdWithAString         *typeCmd1;
  G4UIcmdWithAString         *shapeCmd1;
  G4UIcmdWith3VectorAndUnit  *centreCmd1;
  G4UIcmdWith3Vector         *posrot1Cmd1;
  G4UIcmdWith3Vector         *posrot2Cmd1;
  G4UIcmdWithADoubleAndUnit  *halfxCmd1;
  G4UIcmdWithADoubleAndUnit  *halfyCmd1;
  G4UIcmdWithADoubleAndUnit  *halfzCmd1;
  G4UIcmdWithADoubleAndUnit  *radiusCmd1;
  G4UIcmdWithADoubleAndUnit  *radius0Cmd1;
  G4UIcmdWithADoubleAndUnit  *possigmarCmd1;
  G4UIcmdWithADoubleAndUnit  *possigmaxCmd1;
  G4UIcmdWithADoubleAndUnit  *possigmayCmd1;
  G4UIcmdWithADoubleAndUnit  *paralpCmd1;
  G4UIcmdWithADoubleAndUnit  *partheCmd1;
  G4UIcmdWithADoubleAndUnit  *parphiCmd1;  
  G4UIcmdWithAString         *confineCmd1;  
  
  G4UIcmdWithAString*         relativePlacementCmd;
  G4UIcmdWithAString*         typeCmd ;
  G4UIcmdWithAString*         shapeCmd ;
  G4UIcmdWith3VectorAndUnit*  centreCmd ;
  G4UIcmdWithAString*         positronRangeCmd ;
  G4UIcmdWith3Vector*         posrot1Cmd ;
  G4UIcmdWith3Vector*         posrot2Cmd ;
  G4UIcmdWithADoubleAndUnit*  halfxCmd ;
  G4UIcmdWithADoubleAndUnit*  halfyCmd ;
  G4UIcmdWithADoubleAndUnit*  halfzCmd ;
  G4UIcmdWithADoubleAndUnit*  radiusCmd ;
  G4UIcmdWithADoubleAndUnit*  radius0Cmd ; 
  G4UIcmdWithADoubleAndUnit*  possigmarCmd ;
  G4UIcmdWithADoubleAndUnit*  possigmaxCmd ;
  G4UIcmdWithADoubleAndUnit*  possigmayCmd ;
  G4UIcmdWithADoubleAndUnit*  paralpCmd ;
  G4UIcmdWithADoubleAndUnit*  partheCmd ;
  G4UIcmdWithADoubleAndUnit*  parphiCmd ;  
  G4UIcmdWithAString*         confineCmd ;         

  
  G4UIcmdWithAString         *angtypeCmd1;
  G4UIcmdWithADoubleAndUnit  *angradiusCmd1;
  G4UIcmdWith3VectorAndUnit  *angcentreCmd1;
  G4UIcmdWith3Vector         *angrot1Cmd1;
  G4UIcmdWith3Vector         *angrot2Cmd1;
  G4UIcmdWithADoubleAndUnit  *minthetaCmd1;
  G4UIcmdWithADoubleAndUnit  *maxthetaCmd1;
  G4UIcmdWithADoubleAndUnit  *minphiCmd1;
  G4UIcmdWithADoubleAndUnit  *maxphiCmd1;
  G4UIcmdWithADoubleAndUnit  *angsigmarCmd1;
  G4UIcmdWithADoubleAndUnit  *angsigmaxCmd1;
  G4UIcmdWithADoubleAndUnit  *angsigmayCmd1;
  G4UIcmdWith3VectorAndUnit  *angfocusCmd;
  G4UIcmdWithABool           *useuserangaxisCmd1;
  G4UIcmdWithABool           *surfnormCmd1;


  G4UIcmdWithAString*         angtypeCmd ;
  G4UIcmdWith3Vector*         angrot1Cmd ;
  G4UIcmdWith3Vector*         angrot2Cmd ;
  G4UIcmdWithADoubleAndUnit*  minthetaCmd ;
  G4UIcmdWithADoubleAndUnit*  maxthetaCmd ;
  G4UIcmdWithADoubleAndUnit*  minphiCmd ;
  G4UIcmdWithADoubleAndUnit*  maxphiCmd ;
  G4UIcmdWithADoubleAndUnit*  angsigmarCmd ;
  G4UIcmdWithADoubleAndUnit*  angsigmaxCmd ;
  G4UIcmdWithADoubleAndUnit*  angsigmayCmd ;
  G4UIcmdWithABool*           useuserangaxisCmd ;
  G4UIcmdWithABool*           surfnormCmd ;

  G4UIdirectory              *energyDirectory;
  G4UIcmdWithAString         *energytypeCmd1;
  G4UIcmdWithADoubleAndUnit  *eminCmd1;
  G4UIcmdWithADoubleAndUnit  *emaxCmd1;
  G4UIcmdWithADoubleAndUnit  *monoenergyCmd1;
  G4UIcmdWithADoubleAndUnit  *engsigmaCmd1;
  G4UIcmdWithADouble         *alphaCmd1;
  G4UIcmdWithADouble         *tempCmd1;
  G4UIcmdWithADouble         *ezeroCmd1;
  G4UIcmdWithADouble         *gradientCmd1;
  G4UIcmdWithADouble         *interceptCmd1;
  G4UIcmdWithoutParameter    *calculateCmd1;
  G4UIcmdWithABool           *energyspecCmd1;
  G4UIcmdWithABool           *diffspecCmd1;
 
  
  G4UIcmdWithAString*         energytypeCmd ;
  
  G4UIcmdWithADoubleAndUnit*  eminCmd ;
  G4UIcmdWithADoubleAndUnit*  emaxCmd ;
  G4UIcmdWithADoubleAndUnit*  monoenergyCmd ;
  G4UIcmdWithADoubleAndUnit*  engsigmaCmd ;
  G4UIcmdWithADouble*         alphaCmd ;
  G4UIcmdWithADouble*         tempCmd ;
  G4UIcmdWithADouble*         ezeroCmd ;
  G4UIcmdWithADouble*         gradientCmd ;
  G4UIcmdWithADouble*         interceptCmd ;
  G4UIcmdWithoutParameter*    calculateCmd ;
  G4UIcmdWithABool*           energyspecCmd ;
  G4UIcmdWithABool*           diffspecCmd ;

  G4UIcmdWith3Vector*         histpointCmd1;
  G4UIcmdWithAString*         histtypeCmd;
  G4UIcmdWithAString*         arbintCmd1;
  G4UIcmdWithAString*         resethistCmd1;
  
  
  G4UIcmdWith3Vector*         histpointCmd ;
  G4UIcmdWithAString*         histnameCmd ;
  G4UIcmdWithAString*         arbintCmd ;
  G4UIcmdWithAString*         resethistCmd ;

  G4UIcmdWithAnInteger*       verbosityCmd ;

  // below are commands from G4ParticleGun

  G4UIcommand*                ionCmd ;

  G4UIcmdWithAString*         particleCmd ;
  G4UIcmdWithADoubleAndUnit*  timeCmd ;
  G4UIcmdWith3Vector*         polCmd ;
  G4UIcmdWithAnInteger*       numberCmd ;
  
  G4UIcmdWith3VectorAndUnit*  positionCmd ;
  G4UIcmdWith3Vector*         directionCmd ;
  G4UIcmdWithADoubleAndUnit*  energyCmd ;
  G4UIcmdWithoutParameter*    listCmd ;
  
  G4UIcmdWithAString*         ForbidCmd;
  G4UIcmdWithAString*         setImageCmd1;


  G4UIcmdWithAString*         setUserSpectrumCmd;      

  
 private: // for ion shooting
  
  G4bool   fShootIon ; 
  G4int    fAtomicNumber ;
  G4int    fAtomicMass ;
  G4int    fIonCharge ;
  G4double fIonExciteEnergy ;

};

#endif
