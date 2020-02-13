#ifndef GateLocalTimeResolution_h
#define GateLocalTimeResolution_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

#include "GateVPulseProcessor.hh"

#include "GateMaps.hh"

class GateLocalTimeResolutionMessenger;

class GateLocalTimeResolution : public GateVPulseProcessor
{
  public:

    //! Constructs a new TimeDelay attached to a GateDigitizer
    GateLocalTimeResolution(GatePulseProcessorChain* itsChain,
			       const G4String& itsName) ;

    //! Destructor
    virtual ~GateLocalTimeResolution() ;
    //@}

    //! Adds volume to the hashmap and returns 1 if it exists. If it does not exist, returns 0.

    G4int ChooseVolume(G4String val);

    //! \name setters
    //@{
    //! This function set a TimeDelay for a volume called 'name'.

    void SetTimeResolution(G4String name, G4double val)   { m_table[name].resol = val;  };

    //@}

    //! Implementation of the pure virtual method declared by the base class GateDigitizerComponent
    //! print-out the attributes specific of the TimeDelay
    virtual void DescribeMyself(size_t indent);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:

    struct param {
      G4double resol;
    };

   param m_param;                                 //!<
    G4String m_name;                               //! Name of the volume
    GateMap<G4String,param> m_table ;  //! Table which contains the names of volume with their characteristics
    GateMap<G4String,param> ::iterator im;  //! iterator of the gatemap
    GateLocalTimeResolutionMessenger *m_messenger;       //!< Messenger
};


#endif
