


#ifndef GateEnergyThresholder_h
#define GateEnergyThresholder_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

#include "GateVEffectiveEnergyLaw.hh"
#include "GateDepositedEnergyLaw.hh"
#include "GateSolidAngleWeightedEnergyLaw.hh"
class GateEnergyThresholderMessenger;


class GateEnergyThresholder : public GateVPulseProcessor
{
  public:

    //! Constructs a new EnergyThresholder attached to a GateDigitizer
    GateEnergyThresholder(GatePulseProcessorChain* itsChain,
			       const G4String& itsName, G4double itsThreshold=0) ;
    //! Destructor
    virtual ~GateEnergyThresholder() ;

    //! Returns the threshold
        G4double GetThreshold()   	      { return m_threshold; }

        //! Set the threshold
        void SetThreshold(G4double val)   { m_threshold = val;  }

    //! This function returns the effective energy law in use.
        inline GateVEffectiveEnergyLaw* GetEffectiveEnergyLaw()           { return m_effectiveEnergyLaw; }

        //! This function sets the blurring law for the resolution.
        /*!
          Choose between "linear" and "inverseSquare" blurring law
        */
        inline void SetEffectiveEnergyLaw(GateVEffectiveEnergyLaw* law)   { m_effectiveEnergyLaw = law; }

    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the EnergyThresholder
    virtual void DescribeMyself(size_t indent);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList&  outputPulseList);
    GatePulseList* ProcessPulseList(const GatePulseList* inputPulseList);
  private:

    bool flgTriggerAW;
    std::vector<GateVolumeID> vID;

    GateVEffectiveEnergyLaw* m_effectiveEnergyLaw;
    G4double m_threshold;     	      	      //!< Threshold value
    GateEnergyThresholderMessenger *m_messenger;    //!< Messenger
};


#endif
