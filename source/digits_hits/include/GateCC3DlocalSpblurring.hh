#ifndef GateCC3DlocalSpblurring_h
#define GateCC3DlocalSpblurring_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

#include "GateVPulseProcessor.hh"
#include "G4VoxelLimits.hh"

#include "GateMaps.hh"

class GateCC3DlocalSpblurringMessenger;

/*! \class  GateCC3DlocalSpblurring
    \brief  Pulse-processor for simulating a  spatial gaussian blurring with a different sigma in each spatial direction

    - The user can choose a specific blurring for each tracked volume.

      \sa GateVPulseProcessor
*/
class GateCC3DlocalSpblurring : public GateVPulseProcessor
{
public:

    //! Constructs a new blurring attached to a GateDigitizer
    GateCC3DlocalSpblurring(GatePulseProcessorChain* itsChain,
                            const G4String& itsName) ;

    //! Destructor
    virtual ~GateCC3DlocalSpblurring() ;
    //@}

    //! Adds volume to the hashmap and returns 1 if it exists. If it does not exist, returns 0.

    G4int ChooseVolume(G4String val);

    //! \name setters
    //@{
    //! This function set a blurring for a volume called 'name'.

    
    inline void SetSigmaSpBlurring(G4String name, const G4ThreeVector& val)
    { m_table[name].sigmaSpblurr = val; }

    //@}

    //! Implementation of the pure virtual method declared by the base class GateDigitizerComponent
    //! print-out the attributes specific of the blurring
    virtual void DescribeMyself(size_t indent);

protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

private:

    struct param {
        G4ThreeVector sigmaSpblurr;
    };

    param m_param;                                 //!<
    G4String m_name;                               //! Name of the volume
    GateMap<G4String,param> m_table ;  //! Table which contains the names of volume with their characteristics
    GateMap<G4String,param> ::iterator im;  //! iterator of the gatemap
    GateCC3DlocalSpblurringMessenger *m_messenger;       //!< Messenger

    G4VoxelLimits limits;
    G4double Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
    G4AffineTransform at;
};


#endif
