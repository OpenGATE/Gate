


#ifndef GateDoIModels_h
#define GateDoIModels_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

#include "GateVDoILaw.hh"

class GateDoIModelsMessenger;


class GateDoIModels : public GateVPulseProcessor
{
  public:

    //! Constructs a new EnergyThresholder attached to a GateDigitizer
    //! //I SEE HOW IT WORKS BUT I DO NOT UNDERSTAND : ASK
    /// it loads correcty messenger values
     GateDoIModels(GatePulseProcessorChain* itsChain,  const G4String& itsName);
      ///Like that it   load correcty messenger values but after constuctor
    //GateDoIModels(GatePulseProcessorChain* itsChain,  const G4String& itsName, const G4ThreeVector& itsDoIAxis=G4ThreeVector(0.,0.,1.)) ;
    ///The setfunction call by the messegner load the value inserted in the constructor not the one in the mac
    //GateDoIModels(GatePulseProcessorChain* itsChain,  const G4String& itsName, const G4ThreeVector itsDoIAxis=G4ThreeVector(0.,0.,1.)) ;
    //! Destructor
    virtual ~GateDoIModels() ;

    //!!Only works for selecting one of hte three ortogonal directions (x, y or Z) not other bases
    //! Returns the threshold
       inline const G4ThreeVector& GetDoIAxis() {return m_DoIaxis;}

        //! Set the threshold
      void SetDoIAxis( G4ThreeVector val);

    //! This function returns the effective energy law in use.
        inline GateVDoILaw* GetDoILaw()           { return m_DoILaw; }

        inline void SetDoILaw(GateVDoILaw* law)   { m_DoILaw = law; }

    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the EnergyThresholder
    virtual void DescribeMyself(size_t indent);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList&  outputPulseList);

  private:


    GateVDoILaw* m_DoILaw;
    G4ThreeVector m_DoIaxis;
    GateDoIModelsMessenger *m_messenger;    //!< Messenger
    bool flgCorrectAxis;

};


#endif
