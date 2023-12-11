/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

/*! \class  GateCrosstalk
    \brief  Digitizer Module for simulating an optical and/or an electronic Crosstalk
    - GateCrosstalk - by Martin.Rey@epfl.ch (dec 2002)

    - Digitizer Module for simulating an optical and/or an electronic Crosstalk
    of the scintillation light between the neighbor crystals:
    if the input Digi arrives in a crystal array, Digis around
    it are created (in the edge and corner neighbor crystals).
    ATTENTION: this module functions only for a chosen volume which is an array repeater !!!

	5/12/23 - added to GND by kochebina@cea.fr but MAYBE NOT PROPERLY TESTED !!!!

      \sa GateVDigitizerModule
*/
#ifndef GateCrosstalk_h
#define GateCrosstalk_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateCrosstalkMessenger.hh"
#include "GateSinglesDigitizer.hh"

class GateArrayParamsFinder;

class GateCrosstalk : public GateVDigitizerModule
{
public:
	//! This function allows to retrieve the current instance of the GateCrosstalk singleton
	/*!
	   If the GateCrosstalk already exists, GetInstance only returns a pointer to this singleton.
	   If this singleton does not exist yet, GetInstance creates it by calling the private
	   GateCrosstalk constructor
	*/
	static GateCrosstalk* GetInstance(GateSinglesDigitizer* itsChain,
			const G4String& itsName,
			G4double itsEdgesFraction, G4double itsCornersFraction);


  GateCrosstalk(GateSinglesDigitizer *digitizer, G4String name, G4double itsEdgesFraction, G4double itsCornersFraction);
  ~GateCrosstalk();
  
  void Digitize() override;

  
  void CheckVolumeName(G4String val);

  //! \name getters and setters
  //@{
  //! This function returns the fraction of the part of energy which goes in the edge crystals.
  G4double GetEdgesFraction()                { return m_edgesCrosstalkFraction; }

  //! This function sets the fraction of the part of energy which goes in the edge crystals.
  void SetEdgesFraction (G4double val)       { m_edgesCrosstalkFraction = val;  }

  //! This function returns the fraction of the part of energy which goes in the corner crystals.
  G4double GetCornersFraction()              { return m_cornersCrosstalkFraction; }

  //! This function sets the fraction of the part of energy which goes in the corner crystals.
  void SetCornersFraction (G4double val)     { m_cornersCrosstalkFraction = val;  }

  //! Return the rest Crosstalk per cent
  G4double GetXTPerCent() { return m_XtalkpCent; };
  //@}

  void DescribeMyself(size_t );

private:
//     //! Find the different parameters of the array of detection :
//     //! The numbers of rows in x, y and z
//     //! The position in this matrix of the hit
//     void FindArrayParams(GateVVolume* anInserter);

//     //! Get the VObjectReapeater from an VObjectInserter (if it isn't the right VObjectInserter return 0)
//     GateVGlobalPlacement* GetRepeater(GateVVolume* anInserter);

//     //! Get the ArrayRepeater from an VObjectReapeater (if it isn't an ArrayRepeater return 0)
//     GateArrayRepeater* GetArrayRepeater(GateVGlobalPlacement* aRepeater);

//     //! Find the different parameters of the input Digi :
//     //! e.g. the position in this array of the hit
//     void FindInputDigiParams(const GateVolumeID* m_volumeID);

   //! Create a new VolumeID for the volume of in the matrix with position \i,\j,\k
   GateVolumeID CreateVolumeID(const GateVolumeID* m_volumeID, G4int i, G4int j, G4int k);

   //! Create a new OutputVolumeID for the volume of in the matrix with position \i,\j,\k
   GateOutputVolumeID CreateOutputVolumeID(const GateVolumeID m_volumeID);

   //! Create a new Digi of an energy of \val * ENERGY of \Digi in the volume in position \i,\j,\k
   GateDigi* CreateDigi(G4double val, const GateDigi* Digi, G4int i, G4int j, G4int k);


protected:
   G4double m_XtalkpCent;                                         //!< Actual Crosstalk per cent of energy
   G4double m_edgesCrosstalkFraction, m_cornersCrosstalkFraction; //!< Coefficient which connects energy to the resolution
   GateCrosstalkMessenger *m_messenger;                           //!< Messenger
   G4String m_volume;                                             //!< Name of the Crosstalk volume
   G4int m_testVolume;                                            //!< Equal to 1 if m_volume is a valid volume name, else 0

   GateArrayParamsFinder* ArrayFinder;
   size_t m_nbX, m_nbY, m_nbZ;                                    //!< Parameters of the matrix of detection
   size_t m_i, m_j, m_k;                                          //!< position \i,\j,\k in the matrix
   size_t m_depth;

private:

  //! Static pointer to the GateCrosstalk singleton
  static GateCrosstalk* theGateCrosstalk;

  GateDigi* m_outputDigi;

  GateCrosstalkMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;


};

#endif








