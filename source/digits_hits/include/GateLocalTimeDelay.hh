#ifndef GateLocalTimeDelay_h
#define GateLocalTimeDelay_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

#include "GateVPulseProcessor.hh"

#include "GateMaps.hh"

class GateLocalTimeDelayMessenger;

/*! \class  GateLocalTimeDelay
    \brief  Pulse-processor for simulating a local TimeDelay 

    - The user can choose a specific TimeDelay for each tracked volume.

      \sa GateVPulseProcessor
*/
class GateLocalTimeDelay : public GateVPulseProcessor
{
  public:

    //! Constructs a new TimeDelay attached to a GateDigitizer
    GateLocalTimeDelay(GatePulseProcessorChain* itsChain,
			       const G4String& itsName) ;

    //! Destructor
    virtual ~GateLocalTimeDelay() ;
    //@}

    //! Adds volume to the hashmap and returns 1 if it exists. If it does not exist, returns 0.

    G4int ChooseVolume(G4String val);

    //! \name setters
    //@{
    //! This function set a TimeDelay for a volume called 'name'.

    void SetDelay(G4String name, G4double val)   { m_table[name].delay = val;  };

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
      G4double delay;
    };

   param m_param;                                 //!<
   //  G4double m_delay;
    G4String m_name;                               //! Name of the volume
    GateMap<G4String,param> m_table ;  //! Table which contains the names of volume with their characteristics
    // GateMap<G4String,G4double> m_table ;  //! Table which contains the names of volume with their characteristics
    GateMap<G4String,param> ::iterator im;  //! iterator of the gatemap
    GateLocalTimeDelayMessenger *m_messenger;       //!< Messenger
};


#endif
