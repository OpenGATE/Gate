/*----------------------
  03/2012
  ----------------------*/


#ifndef GateTriCoincidenceSorter_h
#define GateTriCoincidenceSorter_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "GatePulse.hh"
#include "GateDigitizer.hh"
#include "GateVCoincidencePulseProcessor.hh"
#include "GateRootDefs.hh"

#include "G4ThreeVector.hh"


class GateTriCoincidenceSorterMessenger;

// typedef std::vector<G4ThreeVector*> PositionVector;

class GateTriCoincidenceSorter : public GateVCoincidencePulseProcessor
{
public:
//  typedef unsigned long long int  buffer_t;
  GateTriCoincidenceSorter(GateCoincidencePulseProcessorChain* itsChain,
			     const G4String& itsName);

  virtual ~GateTriCoincidenceSorter();



public:
   inline GatePulseList* FindSinglesPulseList(const G4String& pulseListName)
   {return m_digitizer->FindPulseList(pulseListName);}

   inline void SetSinglesPulseListName(G4String& name)      {m_sPulseListName = name ;}

   inline G4double GetTriCoincWindow()                      { return m_triCoincWindow;}
   inline void SetTriCoincWindow( G4double window)          { m_triCoincWindow = window;}

   inline void  SetWSPulseListSize(G4int size)              { m_waitingSinglesSize = size; }
   inline G4int GetWSPulseListSize() const                  { return m_waitingSinglesSize; }

   inline bool IsTriCoincProcessor() const { return 1; }

  virtual void DescribeMyself(size_t indent);
  virtual void RegisterTCSingles(GatePulseList& sPulseList);
  virtual G4String SetSinglesTreeName(const G4String& name);
  virtual void CollectSingles();

protected:

  /*! Implementation of the pure virtual method declared by the base class GateVCoincidencePulseProcessor*/
  GateCoincidencePulse* ProcessPulse(GateCoincidencePulse* inputPulse,G4int iPulse);


private:
   GateDigitizer* m_digitizer;
   G4String m_sPulseListName;
   GateTriCoincidenceSorterMessenger *m_messenger;    //!< Messenger
/*   bool                         m_isTriCoincProc;*/
   G4double m_triCoincWindow;
   GatePulseList*               m_waitingSingles;
   GateRootSingleBuffer         m_sBuffer;
   GateSingleTree*              m_sTree;
   G4String                     m_sTreeName;
   Int_t                        m_triCoincID;
   TBranch*                     m_triCID;
   G4int                        m_waitingSinglesSize;
};

#endif
