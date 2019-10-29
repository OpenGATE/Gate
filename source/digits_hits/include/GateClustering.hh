

#ifndef GateClustering_h
#define GateClustering_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"
//#include "G4Types.hh"
#include "GateMaps.hh"
#include <set>

#include "GateVPulseProcessor.hh"

class GateClusteringMessenger;


class GateClustering : public GateVPulseProcessor
{
public:

    //! Constructs a new pulse-adder attached to a GateDigitizer
    GateClustering(GatePulseProcessorChain* itsChain,const G4String& itsName);

    //! Destructor
    virtual ~GateClustering();



    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the pulse adder
    virtual void DescribeMyself(size_t indent);

    //! Set the threshold
    void SetAcceptedDistance(G4double val) {  m_acceptedDistance = val;  };
    void SetRejectionFlag(G4bool flgval){m_flgMRejection=flgval;};


protected:

    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);


    GatePulseList* ProcessPulseList(const GatePulseList* inputPulseList);





    GateClusteringMessenger *m_messenger;     //!< Messenger


    double getDistance(  G4ThreeVector pos1,G4ThreeVector pos2 );

    void checkClusterCentersDistance( GatePulseList& outputPulseList);
    //std::vector<int > index4Clusters;
    //several functions needed for special processing of electronic pulses
    void PulsePushBack(const GatePulse* inputPulse, GatePulseList& outputPulseList);

    bool same_volumeID(const GatePulse* pulse1, const GatePulse* pulse2 );
private:
    G4double m_acceptedDistance;
    G4bool m_flgMRejection;


};


#endif
