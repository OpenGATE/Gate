

#ifndef GatePulseAdderComptPhotIdealLocal_h
#define GatePulseAdderComptPhotIdealLocal_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"
#include "G4Types.hh"

#include "GateVPulseProcessor.hh"

class GatePulseAdderComptPhotIdealLocalMessenger;


class GatePulseAdderComptPhotIdealLocal : public GateVPulseProcessor
{
  public:

    //! Constructs a new pulse-adder attached to a GateDigitizer
    GatePulseAdderComptPhotIdealLocal(GatePulseProcessorChain* itsChain,const G4String& itsName);

    //! Destructor
    virtual ~GatePulseAdderComptPhotIdealLocal();

    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the pulse adder
    virtual void DescribeMyself(size_t indent);

	//redefined because need to call repack() after last pulse
    GatePulseList* ProcessPulseList(const GatePulseList* inputPulseList);


    //! Adds volume to the hashmap and returns 1 if it exists. If it does not exist, returns 0.
      G4int ChooseVolume(G4String val);
      void SetVolumeName(G4String name) {
             G4cout<<"seting m_name Volume "<<name<<G4endl;
             m_name=name;};

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:
    G4String m_name;                               //! Name of the volume
    GatePulseAdderComptPhotIdealLocalMessenger *m_messenger;     //!< Messenger

   // std::vector<G4int> lastTrackID;
     std::vector<GatePulse> primaryPulsesVol;
     std::vector<G4int> indexPrimVInOut;

     //Thes vector will be only to be able to use EDepMax ( which willl not be a big change)
     std::vector<GatePulse> primaryPulses;
     std::vector<G4int> indexPrimVInPrim;
     std::vector<double> EDepmaxPrimV;

	//several functions needed for special processing of electronic pulses
	void PulsePushBack(const GatePulse* inputPulse, GatePulseList& outputPulseList);
	//void DifferentVolumeIDs(const GatePulse* InputPulse, GatePulseList& outputPulseList);
	//void repackLastVolumeID(GatePulseList& outputPulseList);
     constexpr static double epsilonEnergy=0.00001;//MeV

};


#endif
