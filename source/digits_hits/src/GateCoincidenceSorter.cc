/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "Randomize.hh"

#include "GateCoincidenceSorter.hh"

#include "G4UnitsTable.hh"

#include "GateVolumeID.hh"
#include "GateObjectStore.hh"


#include "GateCoincidenceSorterMessenger.hh"

#include "GateTools.hh"
#include "GateDigitizer.hh"
#include "GateVSystem.hh"
#include "GateCoincidenceDigiMaker.hh"

#include <map>
//------------------------------------------------------------------------------------------------------
G4int GateCoincidenceSorter::gm_coincSectNum=0;
// Constructs a new coincidence sorter, attached to a GateDigitizer and to a system
GateCoincidenceSorter::GateCoincidenceSorter(GateDigitizer* itsDigitizer,
      	      	      	      	      	     const G4String& itsOutputName,
      	      	      	      	      	     G4double itsWindow,
      	      	      	      	      	     const G4String& itsInputName)
  : GateClockDependent(itsDigitizer->GetObjectName()+"/"+itsOutputName),
    m_digitizer(itsDigitizer),
    m_system(0),
    m_outputName(itsOutputName),
    m_inputName(itsInputName),
    m_coincidenceWindow(itsWindow),
    m_coincidenceWindowJitter(0.),
    m_offset(0.),
    m_offsetJitter(0.),
    m_minSectorDifference(2),
    m_depth(1),
    m_coincidentPulses(0),
    m_waitingPulses(0),
    m_isCurrentFinished (false),
    m_multiplesPolicy(kKeepIfAllAreGoods),
    m_allPulseOpenCoincGate(false)
{
  // Create the messenger
  m_messenger = new GateCoincidenceSorterMessenger(this);

  itsDigitizer->InsertDigiMakerModule( new GateCoincidenceDigiMaker(itsDigitizer, itsOutputName,true) );
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Destructor
GateCoincidenceSorter::~GateCoincidenceSorter()
{
  if (m_coincidentPulses)
    delete m_coincidentPulses;
  if (m_waitingPulses)
    delete m_waitingPulses;
  delete m_messenger;
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Oevrload of the virtual method declared by the base class GateVCoincidenceSorter
// print-out the attributes specific of the sorter
void GateCoincidenceSorter::Describe(size_t indent)
{
  GateClockDependent::Describe(indent);
  G4cout << GateTools::Indent(indent) << "Coincidence window: " << G4BestUnit(m_coincidenceWindow,"Time") << G4endl;
  G4cout << GateTools::Indent(indent) << "Coincidence window jitter: " << G4BestUnit(m_coincidenceWindowJitter,"Time") << G4endl;
  G4cout << GateTools::Indent(indent) << "Coincidence offset: " << G4BestUnit(m_offset,"Time") << G4endl;
  G4cout << GateTools::Indent(indent) << "Coincidence offset jitter: " << G4BestUnit(m_offsetJitter,"Time") << G4endl;
  G4cout << GateTools::Indent(indent) << "Min sector diff.:   " << m_minSectorDifference << G4endl;
  G4cout << GateTools::Indent(indent) << "Input:              '" << m_inputName << "'" << G4endl;
  G4cout << GateTools::Indent(indent) << "Output:             '" << m_outputName << "'" << G4endl;
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
void GateCoincidenceSorter::SetMultiplesPolicy(const G4String& policy)
{
    if (policy=="takeWinnerOfGoods")
    	m_multiplesPolicy=kTakeWinnerOfGoods;
    else if (policy=="takeWinnerIfIsGood")
    	m_multiplesPolicy=kTakeWinnerIfIsGood;
    else if (policy=="takeWinnerIfAllAreGoods")
    	m_multiplesPolicy=kTakeWinnerIfAllAreGoods;
    else if (policy=="killAll")
    	m_multiplesPolicy=kKillAll;
    else if (policy=="takeAllGoods")
    	m_multiplesPolicy=kTakeAllGoods;
    else if (policy=="killAllIfMultipleGoods")
    	m_multiplesPolicy=kKillAllIfMultipleGoods;
    else if (policy=="keepIfAnyIsGood")
    	m_multiplesPolicy=kKeepIfAnyIsGood;
    else if (policy=="keepIfOnlyOneGood")
    	m_multiplesPolicy=kKeepIfOnlyOneGood;
    else if (policy=="keepAll")
    	m_multiplesPolicy=kKeepAll;
    else {
    	if (policy!="keepIfAllAreGoods")
    	    G4cout<<"WARNING : policy not recognised, using default : keepMultiplesIfAllAreGoods"<<G4endl;
  	m_multiplesPolicy=kKeepIfAllAreGoods;
    }
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Implementation of the pure virtual method declared by the base class GateVCoincidenceSorter
// Processes a list of pulses and tries to compute a coincidence pulse
void GateCoincidenceSorter::ProcessSinglePulseList(GatePulseList* inp)
{
//   if (m_coincidentPulses){
//     	G4cout<<"ATTENTION0 "
//     	<<m_coincidentPulses->size()<<" current initiales"
// 	<<G4endl;
// 	if (inp)
// 	    G4cout<<"Avec inp a "<<inp->size()<<G4endl;
// 	else
// 	    G4cout<<"Sans inp"<<G4endl;
//   }
//   if (m_waitingPulses){
//     	G4cout<<"ATTENTION0' "
//     	<<m_waitingPulses->size()<<" attente initiales"
// 	<<G4endl;
//   }
  // Check whether we're enabled
  if (!IsEnabled())
    return ;


  // C. Comtat, 07.02.2006
  if (m_offset > 0.0 && m_offset/s < MIN_COINC_OFFSET) {
    G4cout << "Delayed coincidences offset (" << m_offset/s << " sec) should be bigger or equal to " << MIN_COINC_OFFSET << " sec" << G4endl;
    G4Exception( "GateCoincidenceSorter::ProcessSinglePulseList", "ProcessSinglePulseList", FatalException, "Goodbye\n");
  } else if (m_coincidenceWindow/s > MIN_COINC_OFFSET/4) {
    G4cout << "Coincidence window (" << m_coincidenceWindow/s << " sec) should be smaller or equal to " << MIN_COINC_OFFSET/4 << " sec" << G4endl;
    G4Exception( "GateCoincidenceSorter::ProcessSinglePulseList", "ProcessSinglePulseList", FatalException, "Goodbye\n");
  }

  GatePulseList* inputPulseList = inp ? inp : m_digitizer->FindPulseList( m_inputName );

  // Input pulse-list vector is null
  if (!inputPulseList)
    return ;

  // Get the number of input pulses
  G4int n_pulses = inputPulseList->size();
  if (nVerboseLevel==1)
      	G4cout << "[GateCoincidenceSorter::ProcessPulseList]: processing input list with " << inputPulseList->size() << " entries\n";
  if (!n_pulses)
    return ;

  m_isCurrentFinished=false;
  // If there is no coincident pulses and we had a list of waiting pulses, they form the new coincident pulse
  if ( !m_coincidentPulses )
    InitCoincidencePulses(inputPulseList);
  else
    VerifCoincidencePulses(inputPulseList);
  // All incoming pulses are dispatched either into the coincident or into the waiting pulses
  DispatchPulses(inputPulseList);

  if ( !m_isCurrentFinished ) {
    // No pulse is after offset + window, it means that all new pulses are within the coincidence
    // window of the current pulse, and it is still too early to return anything: stop there
    if (nVerboseLevel>=1)
      	G4cout << "[GateCoincidenceSorter::ProcessPulseList]: still in the coincidence window --> returning 0\n";
    return ;
  }


  // At least one of the new pulses is after the coincidence window of the current pulse.
  // We can analyse the current coincidence pulse

  GateCoincidencePulse* outputPulse = m_coincidentPulses;
  m_coincidentPulses = 0;
  G4double stime = outputPulse->ComputeStartTime();
  if (m_allPulseOpenCoincGate){
    for (size_t k1=0;k1<outputPulse->size();k1++){
    	if ((*outputPulse)[k1]->GetTime()>stime){
	    if (!m_waitingPulses) m_waitingPulses = CreateNewCoincidencePulse();
    	    m_waitingPulses->InsertUniqueSortedCopy((*outputPulse)[k1]);
	  //  G4cout<<outputPulse->size()<<"-----------"<<m_waitingPulses<<"----------"<<(*outputPulse)[k1]<<"------"<<(*outputPulse)[k1]->GetTime()<<"---------"<<stime<<G4endl;

	}
    }
  }

  if ((m_multiplesPolicy!=kKeepAll) && (outputPulse->size()<=2)){
    if (CoincidentPulseIsValid(outputPulse))
    	m_digitizer->StoreCoincidencePulse(outputPulse);
    else
    	delete outputPulse;
  } else {
//      GatePulse* firstOfCoinc = outputPulse->FindFirstPulse();
      if (m_multiplesPolicy==kKeepIfAllAreGoods){
	if (CoincidentPulseIsValid(outputPulse)) m_digitizer->StoreCoincidencePulse(outputPulse);
	else delete outputPulse;
      } else if (m_multiplesPolicy==kKeepIfAnyIsGood){
	if (CoincidentPulseIsValid(outputPulse,true)) m_digitizer->StoreCoincidencePulse(outputPulse);
	else delete outputPulse;
      } else if (m_multiplesPolicy==kKeepIfOnlyOneGood){
	GateCoincidencePulse* toRegister=FindIfOnlyOneGood(outputPulse);
	if (toRegister){
//    	    outputPulse->SetTime(toRegister->GetTime());
    	    m_digitizer->StoreCoincidencePulse(outputPulse);
	    delete toRegister;
	} else delete outputPulse;
      } else if (m_multiplesPolicy==kKeepAll){
	if ( (outputPulse->size()>=2) ) {
	  if (nVerboseLevel>=1)
	    G4cout << "[GateCoincidenceSorter::ProcessPulseList]: adding coincidence pulse " << G4endl
		   << *outputPulse << G4endl;
    	    if ( (outputPulse->size()==2) && !CoincidentPulseIsValid(outputPulse))
	    	outputPulse->push_back( new GatePulse((*outputPulse)[0])); // to ensure that the coinc is killed by multiple killer
    	    m_digitizer->StoreCoincidencePulse(outputPulse);
	} else delete outputPulse;
      } else if (m_multiplesPolicy==kKillAll){
	delete outputPulse;
      } else if (m_multiplesPolicy==kKillAllIfMultipleGoods){
	GateCoincidencePulse* toRegister=FindIfOnlyOneGood(outputPulse);
	if ( toRegister) {
	  if (nVerboseLevel>=1)
	    G4cout << "[GateCoincidenceSorter::ProcessPulseList]: adding coincidence pulse " << G4endl
		   << *toRegister << G4endl;
	    m_digitizer->StoreCoincidencePulse(toRegister);
	}
	delete outputPulse;
      } else {
//          G4cout<<"creation sous coinc "<<G4endl;
	  GateCoincidencePulse* toRegister=0;
	  for (size_t k1=0;k1<outputPulse->size();k1++){
	    for (size_t k2=k1+1;k2<outputPulse->size();k2++){
    		GateCoincidencePulse* gp = new GateCoincidencePulse(
		    outputPulse->GetListName(),outputPulse->GetWindow(),outputPulse->GetOffset());


    		if ( (*outputPulse)[k1]->GetTime() < (*outputPulse)[k2]->GetTime() ){
	    	    gp->push_back( new GatePulse((*outputPulse)[k1]));
		    gp->SetStartTime((*outputPulse)[k1]->GetTime());
		    if (gp->IsInCoincidence((*outputPulse)[k2]))
			gp->push_back( new GatePulse((*outputPulse)[k2]));
		} else {
	    	    gp->push_back( new GatePulse((*outputPulse)[k2]));
		    gp->SetStartTime((*outputPulse)[k2]->GetTime());
		    if (gp->IsInCoincidence((*outputPulse)[k1]))
			gp->push_back( new GatePulse((*outputPulse)[k1]));
		}
//                G4cout<<"creation sous coinc termine"<<G4endl;
    		if (gp->size()==2){
		    if (m_multiplesPolicy==kTakeAllGoods ){
		    	if (CoincidentPulseIsValid(gp))
			    m_digitizer->StoreCoincidencePulse(gp);
			else delete gp;
    		    } else if ((m_multiplesPolicy==kTakeWinnerIfIsGood) || (m_multiplesPolicy==kTakeWinnerIfAllAreGoods)){
			if (!toRegister || toRegister->ComputeEnergy()<gp->ComputeEnergy()){
			    if (toRegister) delete toRegister;
			    toRegister=gp;
			} else delete gp;
		    } else {
			if ( CoincidentPulseIsValid(gp) ) {
			  if (nVerboseLevel>=1)
			    G4cout << "[GateCoincidenceSorter::ProcessPulseList]: adding coincidence pulse " << G4endl
				   << *gp << G4endl;
    	    		  if (m_multiplesPolicy==kTakeWinnerOfGoods){
			      if (!toRegister || toRegister->ComputeEnergy()<gp->ComputeEnergy()){
		    		  if (toRegister) delete toRegister;
		    		  toRegister=gp;
			      } else delete gp;
			  } else G4cerr<<"[GateCoincidenceSorter::ProcessOnePulse] Unknown policy situation..."<<G4endl;
			} else {
			    delete gp;
			}
		    }
		} else delete gp;
	    }
	  }

	  if (toRegister) {
    	    G4bool doRegister=true;
      	    if (m_multiplesPolicy==kTakeWinnerIfIsGood)
		doRegister = CoincidentPulseIsValid(toRegister);
            if (m_multiplesPolicy==kTakeWinnerIfAllAreGoods)
		doRegister = CoincidentPulseIsValid(outputPulse);
      	    if (doRegister)
		m_digitizer->StoreCoincidencePulse(toRegister);
	    else
		delete toRegister;
	  }
	  delete outputPulse;
      }
  }


  if (m_waitingPulses){
    GatePulseList* wasWaiting = m_waitingPulses;
    m_waitingPulses=0;
    ProcessSinglePulseList(wasWaiting);
    delete wasWaiting;
  }
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Initialise the coincident pulses as needed
void GateCoincidenceSorter::InitCoincidencePulses(const GatePulseList* inputPulseList)
{
  G4double startTime;

  m_coincidentPulses = CreateNewCoincidencePulse();
  m_isCurrentFinished=false;
  if (m_waitingPulses) {
    // If we had a list of waiting pulses, the first of them
    // form the new current pulse list
    // and we dispatch all of the following pulses into
    // the new current pulse or a new waiting pulse list

    startTime = std::min(m_waitingPulses->ComputeStartTime(),inputPulseList->ComputeStartTime());
    m_coincidentPulses->SetStartTime(startTime);
    GatePulseList* wasWaiting = m_waitingPulses;
    m_waitingPulses = 0;
    DispatchPulses(wasWaiting);
    delete wasWaiting;
  } else {
    // There were no waiting pulses: we create a new one and set its start-time
    startTime = inputPulseList->ComputeStartTime();
    m_coincidentPulses->SetStartTime( startTime );
  }



  if (nVerboseLevel>1)
    G4cout << "[GateCoincidenceSorter::InitCoincidencePulses]: start time set to: " << G4BestUnit(startTime,"Time") << G4endl;
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Verify the coincident pulses as needed
void GateCoincidenceSorter::VerifCoincidencePulses(const GatePulseList* inputPulseList)
{
    GatePulse* firstNew = inputPulseList->FindFirstPulse();
    if (firstNew->GetTime()<m_coincidentPulses->GetStartTime()){
    	m_coincidentPulses->SetStartTime(firstNew->GetTime());
	typedef std::vector< GatePulseList::iterator >  iteratorVect ;
	iteratorVect toWait, toCurrent;
    	for ( GatePulseList::iterator iter = m_coincidentPulses->begin()
	    ; iter != m_coincidentPulses->end()
	    ; ++iter) {
    	    if ( !m_coincidentPulses->IsInCoincidence(*iter) ) {
	    	toWait.push_back(iter);
    	    }
	}
	if (m_waitingPulses){
    	    for ( GatePulseList::iterator iter = m_waitingPulses->begin()
		; iter != m_waitingPulses->end()
		; ++iter) {
    		if ( m_coincidentPulses->IsInCoincidence(*iter) ) {
	    	    toCurrent.push_back(iter);
    		}
	    }
	    for (iteratorVect::reverse_iterator it = toCurrent.rbegin() ; it != toCurrent.rend() ; ++it){
		GatePulse* pulse = **it;
		m_waitingPulses->erase( *it );
		m_coincidentPulses->push_back ( pulse );
	    }
	} else {
	    if (!toWait.empty()) m_waitingPulses = CreateNewCoincidencePulse();
	}
	for (iteratorVect::reverse_iterator it = toWait.rbegin() ; it != toWait.rend() ; ++it){
	    GatePulse* pulse = **it;
	    m_coincidentPulses->erase( *it );
	    m_waitingPulses->push_back ( pulse );
	}
	m_isCurrentFinished=false;
	if (m_waitingPulses && m_waitingPulses->empty()) {delete m_waitingPulses; m_waitingPulses=0;}
    	if (m_waitingPulses){
	    for ( GatePulseList::const_iterator iter = m_waitingPulses->begin()
	    	; iter != m_waitingPulses->end()
	    	; ++iter) {
	     	    if ( m_coincidentPulses->IsAfterWindow(*iter) ) m_isCurrentFinished=true;
    	    }
	    m_waitingPulses->SetStartTime(m_waitingPulses->ComputeStartTime());
	}
    }
    return;

}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Dispatch all pulses either into the coincident or into the waiting pulse lists
void GateCoincidenceSorter::DispatchPulses(const GatePulseList* inputPulseList)
{
  // All incoming pulses go either into the coincident or into the waiting pulses
  for ( GatePulseList::const_iterator iter = inputPulseList->begin(); iter < inputPulseList->end() ; ++iter)
  {
    if ( m_coincidentPulses->IsInCoincidence(*iter) ) {
      	// The incoming pulse is in coincidence with the 'current' pulse: store it there
      	m_coincidentPulses->push_back( new GatePulse(**iter) );
      	if (nVerboseLevel>1)
      	  G4cout << "[GateCoincidenceSorter::ProcessPulseList]: appended new pulse into current coincidence pulse: " << G4endl
	      	 << **iter << G4endl;
    } else {
       // The incoming pulse is not in coincidence with the 'current' pulse: store it in the 'next' coincident pulse

        // There is no waiting pulse list: we create a new one
       if ( !m_waitingPulses ) {
         m_waitingPulses = CreateNewCoincidencePulse();
       }
       // Store the pulse
       m_waitingPulses->push_back( new GatePulse(**iter) );
       // if the time of the pulse is AFTER (and not before) the current pulse list
       // start time, one can mark the current pulse list as completely filled
       if (m_coincidentPulses->IsAfterWindow(*iter) ) m_isCurrentFinished=true;
       if (nVerboseLevel>1)
      	  G4cout << "[GateCoincidenceSorter::ProcessPulseList]: appended new pulse into waiting pulse list: " << G4endl
	      	 << **iter << G4endl;
    }
  }
  m_coincidentPulses->SetStartTime(m_coincidentPulses->ComputeStartTime());
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
inline GateCoincidencePulse* GateCoincidenceSorter::CreateNewCoincidencePulse() const
{
   G4double window = G4RandGauss::shoot(m_coincidenceWindow,m_coincidenceWindowJitter);
   G4double offset = G4RandGauss::shoot(m_offset,m_offsetJitter);
//   G4cout<<"Window "<<window<<" offset "<<offset<<G4endl;
   return new GateCoincidencePulse(m_outputName,window,offset);
}



// Check the validity of the coincident pulse
G4bool GateCoincidenceSorter::CoincidentPulseIsValid(GateCoincidencePulse* outputPulse,G4bool any)
{
  // Check whether we got at least two singles in the coincidence pulse
  if (outputPulse->size() < 2 ) {
    // Nop, we don't have a coincidence: erase and return 0
    if (nVerboseLevel>=1)
        G4cout << "[GateCoincidenceSorter::CoincidentPulseIsValid]: deleting coincidence pulse with only 1 single " << G4endl
	       << *outputPulse << G4endl;
    return false;
  }

  // Look for forbidden coincidences
  for ( GateCoincidencePulse::iterator iter1 = outputPulse->begin(); iter1 < outputPulse->end() ; iter1++)
    for ( GateCoincidencePulse::iterator iter2 = (iter1+1) ; iter2 < outputPulse->end() ; iter2++) {
      if ( IsForbiddenCoincidence(*iter1,*iter2) )
      {
      	if (nVerboseLevel>=1)
          G4cout << "[GateCoincidenceSorter::ProcessPulseList]: deleting coincidence pulse with forbidden coincidence " << G4endl
	      	 << *outputPulse << G4endl;
      	if (!any) return false;
      } else if (any) return true;
    }

  return !any;
}

GateCoincidencePulse* GateCoincidenceSorter::FindIfOnlyOneGood(GateCoincidencePulse* outputPulse){
  if (outputPulse->size() < 2 ) return 0;
  GateCoincidencePulse* ans = 0;
  for ( GateCoincidencePulse::iterator iter1 = outputPulse->begin(); iter1 < outputPulse->end() ; iter1++) {
    for ( GateCoincidencePulse::iterator iter2 = (iter1+1) ; iter2 < outputPulse->end() ; iter2++) {
      if ( !IsForbiddenCoincidence(*iter1,*iter2) ) {
      	GateCoincidencePulse* pls = new GateCoincidencePulse(
	    	     outputPulse->GetListName()
                    ,outputPulse->GetWindow()
	    	    ,outputPulse->GetOffset());
    	if ( (*iter1)->GetTime() < (*iter2)->GetTime() ){
	    pls->push_back( new GatePulse(*iter1));
	    if (pls->IsInCoincidence(*iter2))
		pls->push_back( new GatePulse(*iter2));
	} else {
	    pls->push_back( new GatePulse(*iter2));
	    if (pls->IsInCoincidence(*iter1))
		pls->push_back( new GatePulse(*iter1));
	}
	if (pls->size() == 2) {
            if (ans) {
		delete ans; delete pls;
		return 0;
	    } else {
      		ans = pls;
            }
	} else delete pls;
      }
    }
   }
   return ans;
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
G4int GateCoincidenceSorter::ComputeSectorID(const GatePulse& pulse)
{
    if (m_depth>=(G4int)pulse.GetOutputVolumeID().size()) {
    	G4cerr<<"[GateCoincidenceSorter::ComputeSectorID]: Requiered depth's too deep, setting it to 1"<<G4endl;
	m_depth=1;
    }
    static std::vector<G4int> gkSectorMultiplier;
    static std::vector<G4int> gkSectorNumber;
    if (gkSectorMultiplier.empty()){
    	// this code is done just one time for perfrormance improving
	// one suppose that the system hierarchy is linear until the desired depth
    	GateSystemComponent* comp = m_system->GetBaseComponent();
	G4int depth=0;
	while (comp){
	    gkSectorNumber.push_back(comp->GetAngularRepeatNumber());
	    if ( (depth<m_depth) && ( comp->GetChildNumber() == 1)   ){
	    	comp = comp->GetChildComponent(0);
		depth++;
	    }
	    else
	    	comp=0;
	}
	gkSectorMultiplier.resize(gkSectorNumber.size());
	gkSectorMultiplier[gkSectorNumber.size()-1] = 1;
	for (G4int i=(G4int)gkSectorNumber.size()-2;i>=0;--i){
	    gkSectorMultiplier[i] = gkSectorMultiplier[i+1] * gkSectorNumber[i+1];
	}
	gm_coincSectNum = gkSectorMultiplier[0];
    }
    G4int ans=0;
    for (G4int i=0;i<=m_depth;i++){
    	G4int x = pulse.GetComponentID(i)%gkSectorNumber[i];
    	ans += x*gkSectorMultiplier[i];
    }
    return ans;
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Check whether a coincidence is invalid: ring difference or sector difference too small...
G4bool GateCoincidenceSorter::IsForbiddenCoincidence(const GatePulse& pulse1,const GatePulse& pulse2)
{
  G4int blockID1 = m_system->GetMainComponentID(pulse1),
        blockID2 = m_system->GetMainComponentID(pulse2);

 // Modif by D. Lazaro, February 25th, 2004
  // Computation of sectorID, sectorNumber and sectorDifference, paramaters depending on
  // the geometry construction of the scanner (spherical for system ecatAccel and cylindrical
  // for other systems as Ecat, CPET and cylindricalPET)

  const G4String name = m_system->GetName();
  G4String nameComp = "systems/ecatAccel";
  //G4cout << "NAME OF THE SYSTEM: " << name << "; NAME TO COMPARE: " << nameComp << G4endl;
  int comp = strcmp(name,nameComp);

  if (comp == 0) {
  // Compute the sector difference
  	G4int sectorID1 = m_system->ComputeSectorIDSphere(blockID1),
      	      sectorID2 = m_system->ComputeSectorIDSphere(blockID2);

        // Get the number of sectors per ring
  	G4int sectorNumber = m_system->GetCoincidentSectorNumberSphere();

  	// Deal with the circular difference problem
  	G4int sectorDiff1 = sectorID1 - sectorID2;
  	if (sectorDiff1<0)
    		sectorDiff1 += sectorNumber;
  	G4int sectorDiff2 = sectorID2 - sectorID1;
  	if (sectorDiff2<0)
    		sectorDiff2 += sectorNumber;
 	 G4int sectorDifference = std::min(sectorDiff1,sectorDiff2);

  	//Compare the sector difference with the minimum differences for valid coincidences
  	if (sectorDifference<m_minSectorDifference) {
      	if (nVerboseLevel>1)
      	    G4cout << "[GateCoincidenceSorter::IsForbiddenCoincidence]: coincidence between neighbour blocks --> refused\n";
	return true;
	}
	return false;
  }
  else {
  // Compute the sector difference
  G4int sectorID1 = ComputeSectorID(pulse1),
      	sectorID2 = ComputeSectorID(pulse2);

  // Get the number of sectors per ring
  G4int sectorNumber = GetCoincidentSectorNumber();
  // Deal with the circular difference problem
  G4int sectorDiff1 = sectorID1 - sectorID2;
  if (sectorDiff1<0)
    sectorDiff1 += sectorNumber;
  G4int sectorDiff2 = sectorID2 - sectorID1;
  if (sectorDiff2<0)
    sectorDiff2 += sectorNumber;
  G4int sectorDifference = std::min(sectorDiff1,sectorDiff2);

  //Compare the sector difference with the minimum differences for valid coincidences
  if (sectorDifference<m_minSectorDifference) {
      	if (nVerboseLevel>1)
      	    G4cout << "[GateCoincidenceSorter::IsForbiddenCoincidence]: coincidence between neighbour blocks --> refused\n";
	return true;
  }

  return false;
}
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Next method was added for the multi-system approach

void GateCoincidenceSorter::SetSystem(G4String& inputName)
{
   for (size_t i=0; i<m_digitizer->GetChainNumber() ; ++i)
   {
      G4String pPCOutputName = m_digitizer->GetChain(i)->GetOutputName();

      if(pPCOutputName.compare(inputName) == 0)
      {
         this->SetSystem(m_digitizer->GetChain(i)->GetSystem());
         break;
      }
   }

}
