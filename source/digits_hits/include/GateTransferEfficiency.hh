/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateTransferEfficiency_h
#define GateTransferEfficiency_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

#include "GateVPulseProcessor.hh"

#include "GateMaps.hh"

class GateTransferEfficiencyMessenger;


/*! \class  GateTransferEfficiency
    \brief  Pulse-processor for simulating the transfer efficiency on each crystal.

    - GateTransferEfficiency - by Martin.Rey@epfl.ch (jan 2003)

    - Allows to specify a transfer coefficient for each type of crystals.

      \sa GateVPulseProcessor
*/
class GateTransferEfficiency : public GateVPulseProcessor
{
  public:
    //! This function allows to retrieve the current instance of the GateTransferEfficiency singleton
    /*!
      	If the GateTransferEfficiency already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateTransferEfficiency constructor
    */
    static GateTransferEfficiency* GetInstance(GatePulseProcessorChain* itsChain,
			       const G4String& itsName);

        //! Public Destructor
    virtual ~GateTransferEfficiency() ;

  private:
    //!< Private constructor: this function should only be called from GetInstance()
    GateTransferEfficiency(GatePulseProcessorChain* itsChain,
			       const G4String& itsName);
  public:

    //! Adds volume to the hashmap and returns 1 if it exists. If it does not exist, returns 0.
    G4int ChooseVolume(G4String val);


    //! \name setters and getters
    //@{
    //! Set the transfer efficiency for the crystal called 'name
    void SetTECoeff(G4String name, G4double val)   { m_table[name] = val;  };

    //! Get the light output for crystal called 'name (Nph/MeV)
    G4double GetTECrystCoeff(G4String name)   { return m_table[name]; };

    //! Get the actual transfer efficiency
    G4double GetTECoeff()   { return m_TECoef;  };
    //! Get the minimum transfer efficiency
    G4double GetTEMin();
    //@}

    //! Implementation of the pure virtual method declared by the base class GateDigitizerComponent
    //! print-out the attributes specific of the blurring
    virtual void DescribeMyself(size_t indent);


  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:
    //! Static pointer to the GateTransferEfficiency singleton
    static GateTransferEfficiency* theGateTransferEfficiency;

  private:
    G4String m_name;                               //! Name of the volume
    //! Table which contains the names of volume with their transfert efficiencies
    GateMap<G4String,G4double> m_table ;
    GateMap<G4String,G4double> ::iterator im;
    G4double m_TECoef;        //! Actual transfer efficiency coefficient
    G4double m_TEMin;                  //! Minimum transfer efficiency coefficient

    GateTransferEfficiencyMessenger *m_messenger;       //!< Messenger
};


#endif
