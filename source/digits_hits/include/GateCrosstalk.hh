/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCrosstalk_h
#define GateCrosstalk_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

#include "GateVPulseProcessor.hh"

class GateCrosstalkMessenger;
class GateArrayParamsFinder;
class GateOutputVolumeID;

/*! \class  GateCrosstalk
    \brief  Pulse-processor for simulating an optical and/or an electronic crosstalk
    - GateCrosstalk - by Martin.Rey@epfl.ch (dec 2002)

    - Pulse-processor for simulating an optical and/or an electronic crosstalk
    of the scintillation light between the neighbor crystals:
    if the input pulse arrives in a crystal array, pulses around
    it are created (in the edge and corner neighbor crystals).
    ATTENTION: this module functions only for a chosen volume which is an array repeater !!!


      \sa GateVPulseProcessor
*/
class GateCrosstalk : public GateVPulseProcessor
{
  public:
    //! This function allows to retrieve the current instance of the GateCrosstalk singleton
    /*!
        If the GateCrosstalk already exists, GetInstance only returns a pointer to this singleton.
        If this singleton does not exist yet, GetInstance creates it by calling the private
        GateCrosstalk constructor
    */
    static GateCrosstalk* GetInstance(GatePulseProcessorChain* itsChain,
				      const G4String& itsName,
				      G4double itsEdgesFraction, G4double itsCornersFraction);

        //! Public Destructor
    virtual ~GateCrosstalk() ;

  private:
    //!< Private constructor which Constructs a new crosstalk module attached to a GateDigitizer:
    //! this function should only be called from GetInstance()
    GateCrosstalk(GatePulseProcessorChain* itsChain,
				      const G4String& itsName,
				      G4double itsEdgesFraction, G4double itsCornersFraction) ;

  public:


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

    //! Return the rest crosstalk per cent
    G4double GetXTPerCent() { return m_XtalkpCent; };
    //@}

    //! Implementation of the pure virtual method declared by the base class GateDigitizerComponent
    //! print-out the attributes specific of the crosstalk
    virtual void DescribeMyself(size_t indent);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:
//     //! Find the different parameters of the array of detection :
//     //! The numbers of rows in x, y and z
//     //! The position in this matrix of the hit
//     void FindArrayParams(GateVVolume* anInserter);

//     //! Get the VObjectReapeater from an VObjectInserter (if it isn't the right VObjectInserter return 0)
//     GateVGlobalPlacement* GetRepeater(GateVVolume* anInserter);

//     //! Get the ArrayRepeater from an VObjectReapeater (if it isn't an ArrayRepeater return 0)
//     GateArrayRepeater* GetArrayRepeater(GateVGlobalPlacement* aRepeater);

//     //! Find the different parameters of the input Pulse :
//     //! e.g. the position in this array of the hit
//     void FindInputPulseParams(const GateVolumeID* m_volumeID);

    //! Create a new VolumeID for the volume of in the matrix with position \i,\j,\k
    GateVolumeID CreateVolumeID(const GateVolumeID* m_volumeID, G4int i, G4int j, G4int k);

    //! Create a new OutputVolumeID for the volume of in the matrix with position \i,\j,\k
    GateOutputVolumeID CreateOutputVolumeID(const GateVolumeID m_volumeID);

    //! Create a new Pulse of an energy of \val * ENERGY of \pulse in the volume in position \i,\j,\k
    GatePulse* CreatePulse(G4double val, const GatePulse* pulse, G4int i, G4int j, G4int k);

  private:
    //! Static pointer to the GateCrosstalk singleton
    static GateCrosstalk* theGateCrosstalk;

    G4double m_XtalkpCent;                                         //!< Actual crosstalk per cent of energy
    G4double m_edgesCrosstalkFraction, m_cornersCrosstalkFraction; //!< Coefficient which connects energy to the resolution
    GateCrosstalkMessenger *m_messenger;                           //!< Messenger
    G4String m_volume;                                             //!< Name of the crosstalk volume
    G4int m_testVolume;                                            //!< Equal to 1 if m_volume is a valid volume name, else 0

    GateArrayParamsFinder* ArrayFinder;
    size_t m_nbX, m_nbY, m_nbZ;                                    //!< Parameters of the matrix of detection
    size_t m_i, m_j, m_k;                                          //!< position \i,\j,\k in the matrix
    size_t m_depth;                                                //!< Depth of the selected volume in the Inserter
};


#endif
