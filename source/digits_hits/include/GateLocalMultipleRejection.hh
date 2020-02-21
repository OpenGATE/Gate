#ifndef GateLocalMultipleRejection_h
#define GateLocalMultipleRejection_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

#include "GateVPulseProcessor.hh"

#include "GateMaps.hh"

class GateLocalMultipleRejectionMessenger;

typedef enum {kvolumeName,
              kvolumeID,
             } multiple_def_t;

class GateLocalMultipleRejection : public GateVPulseProcessor
{
  public:

    //! Constructs a new MultipleRejection attached to a GateDigitizer
    GateLocalMultipleRejection(GatePulseProcessorChain* itsChain,
			       const G4String& itsName) ;

    //! Destructor
    virtual ~GateLocalMultipleRejection() ;
    //@}

    //! Adds volume to the hashmap and returns 1 if it exists. If it does not exist, returns 0.

    G4int ChooseVolume(G4String val);

    //! \name setters
    //@{
    //! This function set a MultipleRejection for a volume called 'name'.

    void SetRejectionPolicy(G4String name, G4bool val) { m_table[name].rejectionAllPolicy = val;  }
    void SetMultipleDefinition(G4String name, G4String policy){
        if (policy=="volumeID"){
            m_table[name].multipleDef = kvolumeID;
        }
        else {
            if(policy=="volumeName"){
                m_table[name].multipleDef = kvolumeName;
            }
            else{
                G4cout<<"WARNING : multiple rejection policy not recognized. Default volumeName policy is employed \n";
            }
        }
    }
    //@}

    //! Implementation of the pure virtual method declared by the base class GateDigitizerComponent
    //! print-out the attributes specific of the MultipleRejection
    virtual void DescribeMyself(size_t indent);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);
  GatePulseList* ProcessPulseList(const GatePulseList* inputPulseList);

  private:

    struct param {
        G4bool rejectionAllPolicy;
        multiple_def_t multipleDef;

    };
    std::map<G4String , std::vector<int>> multiplesIndex;
    std::map<G4String , G4bool> multiplesRejPol;

    int currentNumber;
    G4String currentVolumeName;
    param m_param;                                 //!<
    G4String m_name;                               //! Name of the volume
    GateMap<G4String,param> m_table ;  //! Table which contains the names of volume with their characteristics

    GateMap<G4String,param> ::iterator im;  //! iterator of the gatemap
    GateLocalMultipleRejectionMessenger *m_messenger;       //!< Messenger
};


#endif
