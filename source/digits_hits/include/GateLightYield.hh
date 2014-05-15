/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateLightYield_h
#define GateLightYield_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

#include "GateMaps.hh"

class GateLightYieldMessenger;

/*! \class  GateLightYield
    \brief  Pulse-processor for simulating the effect of the light yield of crystal(s).
    You have to give the light yield (LY) of each crystal.

    - GateLightYield - by Martin.Rey@epfl.ch (mai 2003)

    - Pulse-processor for simulating the effect of the light yield of several crystals.
    You have to give the light yield (LY) of each crystal.

      \sa GateVPulseProcessor
*/
class GateLightYield : public GateVPulseProcessor
{

  public:
    //! This function allows to retrieve the current instance of the GateLightYield singleton
    /*!
      	If the GateLightYield already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateLightYield constructor
    */
    static GateLightYield* GetInstance(GatePulseProcessorChain* itsChain,
				       const G4String& itsName);

    //! Public Destructor
    virtual ~GateLightYield() ;

  private:
    //!< Private constructor which Constructs a new blurring module attached to a GateDigitizer:
    //! this function should only be called from GetInstance()
  GateLightYield(GatePulseProcessorChain* itsChain,
		 const G4String& itsName);

  public:

    //! Adds volume to the hashmap and returns 1 if it exists. If it does not exist, returns 0.

    G4int ChooseVolume(G4String val);


    //! \name setters and getters
    //@{

    //! Allows to set the light output for crystal called 'name' (Nph/MeV)
    void SetLightOutput(G4String name, G4double val)   { m_table[name] = val; };

    //! Allows to get the light output for crystal called 'name' (Nph/MeV)
    G4double GetLightOutput(G4String name)   { return m_table[name]; };

    //! Return the actual light output
    G4double GetActLightOutput() { return m_lightOutput; };

    //! Return the minimum light output
    G4double GetMinLightOutput();

    //! Return the crystal name which have the minimum light output
    G4String GetMinLightOutputName()  { return m_minLightOutputName; };
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
    //! Static pointer to the GateLightYield singleton
    static GateLightYield* theGateLightYield;

  private:
    G4String m_name;                               //!< Name of the volume
    GateMap<G4String,G4double> m_table ;  //!< Table which contains the names of volume with their characteristics
    GateMap<G4String,G4double> ::iterator im;  //!< Table iterator
    GateLightYieldMessenger *m_messenger;   //!< Messenger

    G4double m_lightOutput;                        //!< Actual light output

    G4double m_minLightOutput;                     //!< Minimum light output
    G4String m_minLightOutputName;                 //! Crystal name which have the minimum light output
};


#endif
