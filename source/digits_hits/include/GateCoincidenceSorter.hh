/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateCoincidenceSorter_h
#define GateCoincidenceSorter_h 1

#include "globals.hh"
#include <iostream>
#include <list>
#include <deque>
#include "G4ThreeVector.hh"

#include "GateCoincidenceDigi.hh"
#include "GateVDigitizerModule.hh"



class GateCoincidenceSorterMessenger;
class GateVSystem;
class GateDigitizerMgr;

/*! \class  GateCoincidenceSorter
    \brief  Coincidence sorter for a  PET scanner

    - GateCoincidenceSorter - by Daniel.Strul@iphe.unil.ch

    - The sorter processes a series of digis, and stores them into a queue
      When digis get obsolete, it tries to create coincident digi pairs
      It does not rejects multiple coincidences. When successful, it returns a
      coincident digi
*/
//    Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr, for the multi-system approach.

//    01/2016 Rewritten completely by Jared.STRYDHORST@cea.fr

//    2023 Added to GND kochebina@gmail.com

/*
  09/24 kochebina@gmail.com
  Simplification of number of options for multiple coincidences 
  remove "keep" and replace (or remove if double) with "take", 
  i.e. all coincidecnes are now written on disk
  keepIfAllAreGoods = takeWinnerIfAllAreGoods
  keepIfOnlyOneGood = takeWinnerIfOnlyOneGood
  keepIfAnyIsGood = takeWinnerOfGoods
  kKeepAll = removed
*/
typedef enum {kKillAll,
              kTakeAllGoods,
              kKillAllIfMultipleGoods,
              kTakeWinnerOfGoods,
              kTakeWinnerIfIsGood,
              kTakeWinnerIfAllAreGoods,
              kTakeWinnerIfOnlyOneGood
              //kKeepIfOnlyOneGood,
              //kKeepIfAnyIsGood,
              //kKeepAll
              } multiple_policy_t;



typedef enum {kKeepAll_CC,
              kkeepIfMultipleVolumeIDsInvolved_CC,
              kkeepIfMultipleVolumeNamesInvolved_CC} acceptance_policy_4CC_t;

class GateCoincidenceSorter : public GateVDigitizerModule
{
public:

    //! Constructs a new coincidence sorter, attached to a GateDigitizerMgr amd to a system
    GateCoincidenceSorter(GateDigitizerMgr* itsDigitizerMgr,
                          const G4String& itsName,
                          const bool &IsCCSorter=false);
    //! Destructor
    virtual ~GateCoincidenceSorter() ;

    //! Overload of the virtual method declared by the base class GateClockDependent
    //! print-out a description of the sorter
    void DescribeMyself(size_t );
    void Describe(size_t);

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
  
   //! Get the force to 0 minimum sector difference for valid coincidences
    inline G4bool GetForcedTo0MinSectorDifference() const
    { return m_forceMinSecDifferenceToZero; }
    //! Set the force to 0 minimum sector difference for valid coincidences
    inline void SetForcedTo0MinSectorDifference(G4bool diff)
    { m_forceMinSecDifferenceToZero = diff; }

  //! Get the depth of the system-level for coincidences
    inline G4int GetDepth() const
    { return m_depth; }
    //! Set the depth of the system-level for coincidences
    inline void SetDepth(G4int depth)
    { m_depth = depth; }

    inline G4bool GetAllDigiOpenCoincGate() const
    { return m_allDigiOpenCoincGate; }
    inline void SetAllDigiOpenCoincGate(G4bool b)
    { m_allDigiOpenCoincGate = b; }


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
    {   m_inputName = anInputName;}

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
    //! Processes a list of digis and tries to compute a coincidence digi
     void Digitize() override;


    virtual inline GateVSystem* GetSystem() const
    { return m_system;}
    virtual inline void SetSystem(GateVSystem* aSystem)
    { m_system = aSystem; }
    void SetSystem(G4String& inputName); //This method was added for the multi-system approach

    void SetMultiplesPolicy(const G4String& policy);
    //TODO GND 2022 CC
    void SetAcceptancePolicy4CC(const G4String& policy);


    G4double GetMinS() const {return m_minS;}
    G4double GetMaxDeltaZ() const {return m_maxDeltaZ;}

    //! Set the GeometrySelector
    void SetMinS(G4double val) { m_minS = val;}
    void SetMaxDeltaZ(G4double val) { m_maxDeltaZ = val;}



protected:
    //! \name Parameters of the sorter
    //@{

    GateDigitizerMgr       *m_digitizerMgr;
    GateVSystem         *m_system;                      //!< System to which the sorter is attached
    G4String            m_outputName;
    G4String            m_inputName;
    G4double            m_coincidenceWindow;            //!< Coincidence time window
    G4double            m_coincidenceWindowJitter;      //!< Coincidence time window jitter
    G4double            m_offset;                       //!< Offset window
    G4double            m_offsetJitter;                 //!< Offset window jitter
    G4int               m_minSectorDifference;          //!< Minimum sector difference for valid coincidences
    multiple_policy_t   m_multiplesPolicy;              //!< Do what if multiples?
    acceptance_policy_4CC_t m_acceptance_policy_4CC;    //! <Which is the criteria to accept coincidences in CC sys
    G4bool              m_allDigiOpenCoincGate;        //!< can a digi be part of two coincs?
    G4int               m_depth;                        //!< Depth of system-level for coincidences
    G4double m_minS;  //!< contains the rebirth time.
    G4double m_maxDeltaZ;  //!< contains the rebirth time.
    G4int coincID_CC;


    //@}

private:
    //! \name Work storage variable
    //@{
    G4bool m_forceMinSecDifferenceToZero;
 
    std::list<GateDigi*> m_presortBuffer;      // incoming digis are presorted and buffered
    G4int                 m_presortBufferSize;
    G4bool                m_presortWarning;     // avoid repeat warnings
  
    bool                m_CCSorter;     // compton camera sorter
    G4bool             m_triggerOnlyByAbsorber; //! Is the window only open by digis generated in the absorber ?
    G4String      m_absorberSD;// absorber "SD' volume name CC
    G4bool             m_eventIDCoinc; //


    std::deque<GateCoincidenceDigi*> m_coincidenceDigis;  // open coincidence windows

    void ProcessCompletedCoincidenceWindow(GateCoincidenceDigi*);
    //TODO GND 2022 CC
    void ProcessCompletedCoincidenceWindow4CC(GateCoincidenceDigi *);

    G4bool IsForbiddenCoincidence(const GateDigi* digi1,const GateDigi* digi2);
    //TODO GND 2022 CC
    G4bool IsCoincidenceGood4CC(GateCoincidenceDigi* coincidence);
    GateCoincidenceDigi* CreateSubDigi(GateCoincidenceDigi* coincidence, G4int i, G4int j);
    G4int ComputeSectorID(const GateDigi& digi);
    static G4int          gm_coincSectNum;     // internal use


    GateCoincidenceDigiCollection*  m_OutputCoincidenceDigiCollection;

    //@}


    GateCoincidenceSorterMessenger *m_messenger;      //!< Messenger

public:
    G4int m_outputDigiCollectionID;
    G4String            m_coincidenceSorterName;

};


#endif
