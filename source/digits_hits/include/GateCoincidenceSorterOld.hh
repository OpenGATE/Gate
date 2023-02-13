/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateCoincidenceSorterOld_h
#define GateCoincidenceSorterOld_h 1

#include "globals.hh"
#include <iostream>
#include <list>
#include <deque>
#include "G4ThreeVector.hh"

#include "GateCoincidencePulse.hh"
#include "GateClockDependent.hh"



class GateCoincidenceSorterOldMessenger;
class GateVSystem;
class GateDigitizer;

/*! \class  GateCoincidenceSorterOld
    \brief  Coincidence sorter for a  PET scanner

    - GateCoincidenceSorterOld - by Daniel.Strul@iphe.unil.ch

    - The sorter processes a series of pulses, and stores them into a queue
      When pulses get obsolete, it tries to create coincident pulse pairs
      It does not rejects multiple coincidences. When successful, it returns a
      coincident pulse
*/
//    Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr, for the multi-system approach.

//    01/2016 Rewritten completely by Jared.STRYDHORST@cea.fr

typedef enum {kKillAllOld,
              kTakeAllGoodsOld,
              kKillAllIfMultipleGoodsOld,
              kTakeWinnerOfGoodsOld,
              kTakeWinnerIfIsGoodOld,
              kTakeWinnerIfAllAreGoodsOld,
              kKeepIfAllAreGoodsOld,
              kKeepIfOnlyOneGoodOld,
              kKeepIfAnyIsGoodOld,
              kKeepAllOld} multiple_policy_t_old;



typedef enum {kKeepAllOld_CC,
              kkeepIfMultipleVolumeIDsInvolvedOld_CC,
              kkeepIfMultipleVolumeNamesInvolvedOld_CC} acceptance_policy_4CC_t_old;

class GateCoincidenceSorterOld : public GateClockDependent
{
public:

    //! Constructs a new coincidence sorter, attached to a GateDigitizer amd to a system
    GateCoincidenceSorterOld(GateDigitizer* itsDigitizer,
                          const G4String& itsName,
                          G4double itsWindow,
                          const G4String& itsInputName="Singles", const bool &IsCCSorter=false);
    //! Destructor
    virtual ~GateCoincidenceSorterOld() ;

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
    {  m_coincidenceWindow = val; }

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


    inline G4bool GetIfTriggerOnlyByAbsorber() const
    { return m_triggerOnlyByAbsorber; }
    inline void SetIfTriggerOnlyByAbsorber(G4bool b)
    { m_triggerOnlyByAbsorber = b; }


    inline G4bool GetIfEventIDCoinc() const
    { return m_eventIDCoinc; }
    inline void SetIfEventIDCoinc(G4bool b)
    { m_eventIDCoinc = b; }


    const G4String& GetInputName() const
    { return m_inputName; }
    void SetInputName(const G4String& anInputName)
    {  m_inputName = anInputName; }

    const G4String& GetOutputName() const
    { return m_outputName; }

    void SetPresortBufferSize(G4int size)
    { m_presortBufferSize = size; }

    inline void SetAbsorberSDVol(G4String val)
    { m_absorberSD = val;
        // G4cout<<"m_absorDepth2 "<<m_absorberDepth2Name<<G4endl;
    }


    //@}

    //! \name Methods for coincidence sorting
    //@{

    //! Implementation of the pure virtual method declared by our base class
    //! Processes a list of pulses and tries to compute a coincidence pulse
    virtual void ProcessSinglePulseList(GatePulseList* inp=0);


    virtual inline GateVSystem* GetSystem() const
    { return m_system;}
    virtual inline void SetSystem(GateVSystem* aSystem)
    { m_system = aSystem; }
    void SetSystem(G4String& inputName); //This method was added for the multi-system approach

    void SetMultiplesPolicy(const G4String& policy);
    void SetAcceptancePolicy4CC(const G4String& policy);


protected:
    //! \name Parameters of the sorter
    //@{

    GateDigitizer       *m_digitizer;
    GateVSystem         *m_system;                      //!< System to which the sorter is attached
    G4String            m_outputName;
    G4String            m_inputName;
    G4double            m_coincidenceWindow;            //!< Coincidence time window
    G4double            m_coincidenceWindowJitter;      //!< Coincidence time window jitter
    G4double            m_offset;                       //!< Offset window
    G4double            m_offsetJitter;                 //!< Offset window jitter
    G4int               m_minSectorDifference;          //!< Minimum sector difference for valid coincidences
    multiple_policy_t_old   m_multiplesPolicy;              //!< Do what if multiples?
    acceptance_policy_4CC_t_old m_acceptance_policy_4CC;    //! <Which is the criteria to accept coincidences in CC sys
    G4bool              m_allPulseOpenCoincGate;        //!< can a pulse be part of two coincs?
    G4int               m_depth;                        //!< Depth of system-level for coincidences

    G4int coincID_CC;

    //@}

private:
    //! \name Work storage variable
    //@{

    std::list<GatePulse*> m_presortBuffer;      // incoming pulses are presorted and buffered
    G4int                 m_presortBufferSize;
    G4bool                m_presortWarning;     // avoid repeat warnings
    bool                m_CCSorter;     // compton camera sorter
    G4bool             m_triggerOnlyByAbsorber; //! Is the window only open by pulses generated in the absorber ?
    G4String      m_absorberSD;// absorber "SD' volume name CC
    G4bool             m_eventIDCoinc; //


    std::deque<GateCoincidencePulse*> m_coincidencePulses;  // open coincidence windows

    void ProcessCompletedCoincidenceWindow(GateCoincidencePulse*);
    void ProcessCompletedCoincidenceWindow4CC(GateCoincidencePulse *);

    G4bool IsForbiddenCoincidence(const GatePulse* pulse1,const GatePulse* pulse2);
    G4bool IsCoincidenceGood4CC(GateCoincidencePulse* coincidence);
    GateCoincidencePulse* CreateSubPulse(GateCoincidencePulse* coincidence, G4int i, G4int j);
    G4int ComputeSectorID(const GatePulse& pulse);
    static G4int          gm_coincSectNum;     // internal use

    //@}


    GateCoincidenceSorterOldMessenger *m_messenger;      //!< Messenger
};


#endif
