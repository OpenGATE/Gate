

#ifndef GatePulseAdderComptPhotIdeal_h
#define GatePulseAdderComptPhotIdeal_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"
#include "G4Types.hh"

#include "GateVPulseProcessor.hh"

class GatePulseAdderComptPhotIdealMessenger;


class GatePulseAdderComptPhotIdeal : public GateVPulseProcessor
{
  public:

    //! Constructs a new pulse-adder attached to a GateDigitizer
    GatePulseAdderComptPhotIdeal(GatePulseProcessorChain* itsChain,const G4String& itsName);

    //! Destructor
    virtual ~GatePulseAdderComptPhotIdeal();

    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the pulse adder
    virtual void DescribeMyself(size_t indent);

	//redefined because need to call repack() after last pulse
    //GatePulseList* ProcessPulseList(const GatePulseList* inputPulseList);
    std::vector<G4int> lastTrackID;
    bool flgEvtRej;
    void SetEvtRejectionPolicy(G4bool flgval){m_flgRejActPolicy=flgval;};

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);
    GatePulseList* ProcessPulseList(const GatePulseList* inputPulseList);
  private:
    GatePulseAdderComptPhotIdealMessenger *m_messenger;     //!< Messenger

	//several functions needed for special processing of electronic pulses
	void PulsePushBack(const GatePulse* inputPulse, GatePulseList& outputPulseList);
	//void DifferentVolumeIDs(const GatePulse* InputPulse, GatePulseList& outputPulseList);
	//void repackLastVolumeID(GatePulseList& outputPulseList);
    G4bool m_flgRejActPolicy;
     constexpr static double epsilonEnergy=0.00001;//MeV

};


#endif
