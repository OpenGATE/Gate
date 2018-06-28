

#ifndef GateLocalClustering_h
#define GateLocalClustering_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"
//#include "G4Types.hh"
#include "GateMaps.hh"
#include <set>

#include "GateVPulseProcessor.hh"

class GateLocalClusteringMessenger;


class GateLocalClustering : public GateVPulseProcessor
{
  public:

    //! Constructs a new pulse-adder attached to a GateDigitizer
    GateLocalClustering(GatePulseProcessorChain* itsChain,const G4String& itsName);

    //! Destructor
    virtual ~GateLocalClustering();

    //! Adds volume to the hashmap and returns 1 if it exists. If it does not exist, returns 0.
   G4int ChooseVolume(G4String val);


    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the pulse adder
    virtual void DescribeMyself(size_t indent);

    //! Set the threshold
    void SetAcceptedDistance(G4String name, G4double val) {  m_table[name].distance = val;  };
    void SetRejectionFlag(G4String name, G4bool flgval){m_table[name].Rejectionflg=flgval;};


//     void SetVolumeName(G4String name) {
//         G4cout<<"seting m_name Volume "<<name<<G4endl;
//         m_name=name;};

//     G4String GetVolumeName() {
//         return m_name;};



protected:

  //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
  //! This methods processes one input-pulse
  //! It is is called by ProcessPulseList() for each of the input pulses
  //! The result of the pulse-processing is incorporated into the output pulse-list
  void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);
  GatePulseList* ProcessPulseList(const GatePulseList* inputPulseList);



  double getDistance(  G4ThreeVector pos1,G4ThreeVector pos2 );


 std::vector<int > index4Clusters;
  //several functions needed for special processing of electronic pulses
  void PulsePushBack(const GatePulse* inputPulse, GatePulseList& outputPulseList);


private:

  struct param {

      G4double distance;
      G4bool Rejectionflg;
  };
     param m_param;

    //G4String m_name;                               //! Name of the volume
    GateLocalClusteringMessenger *m_messenger;     //!< Messenger
    GateMap<G4String,param> m_table ;  //! Table which contains the names of volume with their characteristics
    GateMap<G4String,param> ::iterator im;  //! iterator of the gatemap





};


#endif
