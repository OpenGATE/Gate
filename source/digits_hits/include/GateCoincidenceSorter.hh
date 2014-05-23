/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCoincidenceSorter_h
#define GateCoincidenceSorter_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateCoincidencePulse.hh"
#include "GateClockDependent.hh"


class GateCoincidenceSorterMessenger;
class GateVSystem;
class GateDigitizer;

/*! \class  GateCoincidenceSorter
    \brief  Coincidence sorter for a  PET scanner

    - GateCoincidenceSorter - by Daniel.Strul@iphe.unil.ch

    - The sorter processes a series of pulses, and stores them into a queue
      When pulses get obsolete, it tries to create coincident pulse pairs
      It does not rejects multiple coincidences. When successful, it returns a
      coincident pulse
*/
//    Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr, for the multi-system approach.

class GateCoincidenceSorter : public GateClockDependent
{
  public:

    //! Constructs a new coincidence sorter, attached to a GateDigitizer amd to a system
    GateCoincidenceSorter(GateDigitizer* itsDigitizer,
                          const G4String& itsName,
                          G4double itsWindow,
                          const G4String& itsInputName="Singles");
    //! Destructor
    virtual ~GateCoincidenceSorter() ;

    //! Overload of the virtual method declared by the base class GateClockDependent
    //! print-out a description of the sorter
    virtual void Describe(size_t indent);

    //! \name getters and setters
    //@{

    //! Get the coincidence time-window
    virtual inline G4double GetWindow() const
        { return m_coincidenceWindow;}

    //! Set the coincidence time-window
    virtual inline void SetWindow(G4double val)
        {
          m_coincidenceWindow = val;
          if (m_coincidentPulses)
            m_coincidentPulses->SetWindow(m_coincidenceWindow);
          if (m_waitingPulses)
            m_waitingPulses->SetWindow(m_coincidenceWindow);
        }
    //! Set the coincidence time-window
    virtual inline void SetWindowJitter(G4double val)
    {  m_coincidenceWindowJitter = val;}
    //! Set the coincidence offset window
    inline void SetOffset(G4double val)
    { m_offset = val;}
    //! Set the coincidence offset window jitter
    inline void SetOffsetJitter(G4double val)
    { m_offsetJitter = val;}
    //! Get the minimum sector difference for valid coincidences
    inline G4int GetMinSectorDifference() const
        { return m_minSectorDifference; }

    //! Set the minimum sector difference for valid coincidences
    inline void SetMinSectorDifference(G4int diff)
        { m_minSectorDifference = diff; }

    //! Get the depth of the system-level for coincidences
    inline G4int GetDepth() const
        { return m_depth; }

    //! Set the depth of the system-level for coincidences
    inline void SetDepth(G4int depth)
        { m_depth = depth; }
    inline G4bool GetAllPulseOpenCoincGate() const
        { return m_allPulseOpenCoincGate; }

    inline void SetAllPulseOpenCoincGate(G4bool b)
        { m_allPulseOpenCoincGate = b; }

     const G4String& GetInputName() const
       { return m_inputName; }
     void SetInputName(const G4String& anInputName)
       {  m_inputName = anInputName; }
     const G4String& GetOutputName() const
       { return m_outputName; }

    //@}

    //! \name Methods for coincidence sorting
    //@{

    //! Implementation of the pure virtual method declared by our base class
    //! Processes a list of pulses and tries to compute a coincidence pulse
    virtual void ProcessSinglePulseList(GatePulseList* inp=0);

    //! Initialise the coincident pulses as needed
    virtual void InitCoincidencePulses(const GatePulseList* inputPulseList);
    //! Verify than the new pulse list has no pulse before the current list
    virtual void VerifCoincidencePulses(const GatePulseList* inputPulseList);

    // Dispatch all pulses either into the coincident or into the waiting pulse lists
    virtual void DispatchPulses(const GatePulseList* inputPulseList);

    // Check the validity of the coincident pulse
    virtual G4bool CoincidentPulseIsValid(GateCoincidencePulse* outputPulse,G4bool any=false);
    // Compute number of sub coinc valid within output
    GateCoincidencePulse* FindIfOnlyOneGood(GateCoincidencePulse* outputPulse);

    //! Check whether a coincidence is invalid: sector difference too small...
    G4bool IsForbiddenCoincidence(const GatePulse& pulse1,const GatePulse& pulse2);

    //@}


     virtual inline GateVSystem* GetSystem() const
       { return m_system;}
     virtual inline void SetSystem(GateVSystem* aSystem)
       { m_system = aSystem; }
    void SetMultiplesPolicy(const G4String& policy);
    void SetSystem(G4String& inputName); //This method was added for the multi-system approach

  protected:
    //! \name Parameters of the sorter
    //@{

    GateDigitizer *m_digitizer;
    GateVSystem *m_system;            //!< System to which the sorter is attached
    G4String     m_outputName;
    G4String     m_inputName;
    G4double m_coincidenceWindow;     //!< Coincidence time window
    G4double m_coincidenceWindowJitter;     //!< Coincidence time window
    G4double m_offset;          //!< Offset window
    G4double m_offsetJitter;          //!< Offset window
    G4int m_minSectorDifference;      //!< Minimum sector difference for valid coincidences
    G4int m_depth;                    //!< Deph of system-level for coincidences

    //@}
    static G4int GetCoincidentSectorNumber() {return gm_coincSectNum;}
    G4int ComputeSectorID(const GatePulse& pulse);
    typedef enum {kKillAll,kTakeAllGoods,kKillAllIfMultipleGoods,kTakeWinnerOfGoods,kTakeWinnerIfIsGood,kTakeWinnerIfAllAreGoods
      ,kKeepIfAllAreGoods,kKeepIfOnlyOneGood,kKeepIfAnyIsGood,kKeepAll} multiple_policy_t;
  private:
    //! \name Work storage variable
    //@{
    GateCoincidencePulse* CreateNewCoincidencePulse() const;
    GateCoincidencePulse* m_coincidentPulses;  //!< Current coincident pulse
    GateCoincidencePulse* m_waitingPulses;     //!< Next coincidence pulse
    G4bool                m_isCurrentFinished; //!< is the current pulse finished
    multiple_policy_t     m_multiplesPolicy;   //!< Do what if multiples?
    G4bool                m_allPulseOpenCoincGate; //! <does a pulse can be part of two coincs?
    static G4int          gm_coincSectNum;     // internal use
    //@}

    GateCoincidenceSorterMessenger *m_messenger;      //!< Messenger
};


#endif
