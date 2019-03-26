


#ifndef GateLocalEnergyThresholder_h
#define GateLocalEnergyThresholder_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

#include "GateMaps.hh"
#include "GateObjectStore.hh"

#include "GateVEffectiveEnergyLaw.hh"
#include "GateDepositedEnergyLaw.hh"
#include "GateSolidAngleWeightedEnergyLaw.hh"
class GateLocalEnergyThresholderMessenger;


class GateLocalEnergyThresholder : public GateVPulseProcessor
{
  public:

    //! Constructs a new EnergyThresholder attached to a GateDigitizer
    GateLocalEnergyThresholder(GatePulseProcessorChain* itsChain,
                   const G4String& itsName) ;
    //! Destructor
    virtual ~GateLocalEnergyThresholder() ;

    //! Adds volume to the hashmap and returns 1 if it exists. If it does not exist, returns 0.
   G4int ChooseVolume(G4String val);

   void SetThreshold(G4String name, G4double val) {  m_table[name].m_threshold = val;  };
   void SetEffectiveEnergyLaw(G4String name, GateVEffectiveEnergyLaw* law) {
       G4cout<<""<<law->GetObjectName()<<G4endl;
       m_table[name].m_effectiveEnergyLaw=law;
      G4cout<<m_table[name].m_effectiveEnergyLaw->GetObjectName()<<G4endl;};

//    //! Returns the threshold
//        G4double GetThreshold()   	      { return m_threshold; }


//    //! This function returns the effective energy law in use.
//        inline GateVEffectiveEnergyLaw* GetEffectiveEnergyLaw()           { return m_effectiveEnergyLaw; }




    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the EnergyThresholder
    virtual void DescribeMyself(size_t indent);

  protected:
  
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList&  outputPulseList);
    GatePulseList* ProcessPulseList(const GatePulseList* inputPulseList);
  private:

    bool flgTriggerAW;
  std::vector<GateVolumeID> vID;


    GateLocalEnergyThresholderMessenger *m_messenger;    //!< Messenger

    struct param {

        G4double m_threshold;
        GateVEffectiveEnergyLaw* m_effectiveEnergyLaw;
    };
       param m_param;
        G4String m_name;                               //! Name of the volume
      GateMap<G4String,param> m_table ;  //! Table which contains the names of volume with their characteristics
      GateMap<G4String,param> ::iterator im;  //! iterator of the gatemap

      //---------------test for multiples close to each other
      // std::map<G4String, std::vector<int>> PulsesIndexBelowSolidAngleTHR;
      //-------------------
};


#endif
