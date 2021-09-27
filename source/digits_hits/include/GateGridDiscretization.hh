

#ifndef GateGridDiscretization_h
#define GateGridDiscretization_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"
//#include "G4Types.hh"
#include "GateMaps.hh"
#include <set>
#include "GateVVolume.hh"
#include "G4VoxelLimits.hh"

#include "GateVPulseProcessor.hh"

class GateGridDiscretizationMessenger;


class GateGridDiscretization : public GateVPulseProcessor
{
public:

    //! Constructs a new pulse-adder attached to a GateDigitizer
    GateGridDiscretization(GatePulseProcessorChain* itsChain,const G4String& itsName);

    //! Destructor
    virtual ~GateGridDiscretization();

    //! Adds volume to the hashmap and returns 1 if it exists. If it does not exist, returns 0.
    G4int ChooseVolume(G4String val);


    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the pulse adder
    virtual void DescribeMyself(size_t indent);

    //! Set the threshold
    // void SetThreshold(G4String name, G4double val) {  m_table[name].threshold = val;  };
    void SetStripOffsetX(G4String name, G4double val)   {  m_table[name].stripOffsetX = val;  };
    void SetStripOffsetY(G4String name, G4double val)   {  m_table[name].stripOffsetY = val;  };
    void SetStripOffsetZ(G4String name, G4double val)   {  m_table[name].stripOffsetZ = val;  };
    //
    void SetNumberStripsX(G4String name,int val)   {  m_table[name].numberStripsX = val;  };
    void SetNumberStripsY(G4String name, int val)   {  m_table[name].numberStripsY = val;  };
    void SetNumberStripsZ(G4String name, int val)   {  m_table[name].numberStripsZ = val;  };
    //
    void SetStripWidthX(G4String name,G4double val)   {  m_table[name].stripWidthX = val;  };
    void SetStripWidthY(G4String name, G4double val)   {  m_table[name].stripWidthY = val;  };
    void SetStripWidthZ(G4String name, G4double val)   {  m_table[name].stripWidthZ = val;  };
    //
    void SetNumberReadOutBlocksX(G4String name,int val)   {  m_table[name].numberReadOutBlocksX = val;  };
    void SetNumberReadOutBlocksY(G4String name, int val)   {  m_table[name].numberReadOutBlocksY = val;  };
    void SetNumberReadOutBlocksZ(G4String name, int val)   {  m_table[name].numberReadOutBlocksZ = val;  };
    // void SetRejectionFlag(G4String name, G4bool flgval){m_table[name].rejectionFlg=flgval;};


    void SetVolumeName(G4String name) {
        G4cout<<"seting m_name Volume "<<name<<G4endl;
        m_name=name;};



protected:

    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    //void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList, std::vector<int> & indexX, std::vector<int>&  indexY);
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

    //private:
public:
    // void ApplyEnergyThreshold( GatePulseList& outputPulseList);
    void ApplyBlockReadOut( GatePulseList& outputPulseList);
    //void ApplyMultipleRejection( GatePulseList& outputPulseList);

    GatePulseList* ProcessPulseList(const GatePulseList* inputPulseList);

    std::vector<int > index_X_list;
    std::vector<int > index_Y_list;
    std::vector<int > index_Z_list;

    //std::map<std::pair<int,int>, std::vector<int>> blockIndex;
    std::map<std::tuple<int,int,int>, std::vector<int>> blockIndex;

    struct param {

        // G4double threshold;
        G4double stripOffsetX;
        G4double stripOffsetY;
        G4double stripOffsetZ;
        G4double stripWidthX;
        G4double stripWidthY;
        G4double stripWidthZ;
        G4int numberStripsX;
        G4int numberStripsY;
        G4int numberStripsZ;
        G4int numberReadOutBlocksX;
        G4int numberReadOutBlocksY;
        G4int numberReadOutBlocksZ;
        //G4bool rejectionFlg;

        G4ThreeVector volSize;
        G4double deadSpX;
        G4double deadSpY;
        G4double deadSpZ;
        G4double pitchY;
        G4double pitchX;
        G4double pitchZ;
    };
    param m_param;

    G4String m_name;                               //! Name of the volume
    GateGridDiscretizationMessenger *m_messenger;     //!< Messenger
    GateMap<G4String,param> m_table ;  //! Table which contains the names of volume with their characteristics
    GateMap<G4String,param> ::iterator im;  //! iterator of the gatemap



    G4VoxelLimits limits;
    G4double min, max;
    G4AffineTransform at;


    void SetGridPoints3D( int indexX, int indexY, int indexZ, G4ThreeVector& pos );

    int GetXIndex(G4double posX);
    int GetYIndex(G4double posY);
    int GetZIndex(G4double posZ);

    //several functions needed for special processing of electronic pulses
    void PulsePushBack(const GatePulse* inputPulse, GatePulseList& outputPulseList);


    int INVALID_INDEX=-2;
    double EPSILON=1e-9;
};


#endif
