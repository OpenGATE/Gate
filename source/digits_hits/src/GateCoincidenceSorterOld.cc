/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

//GND:ClassToRemove

#include "Randomize.hh"

#include "GateCoincidenceSorterOld.hh"

#include "G4UnitsTable.hh"

#include "GateVolumeID.hh"
#include "GateObjectStore.hh"


#include "GateCoincidenceSorterOldMessenger.hh"

#include "GateTools.hh"
#include "GateDigitizer.hh"
#include "GateVSystem.hh"
//#include "GateCoincidenceDigiMaker.hh"

//#include <map>

//------------------------------------------------------------------------------------------------------
G4int GateCoincidenceSorterOld::gm_coincSectNum=0;
// Constructs a new coincidence sorter, attached to a GateDigitizer and to a system
GateCoincidenceSorterOld::GateCoincidenceSorterOld(GateDigitizer* itsDigitizer,
                                             const G4String& itsOutputName,
                                             G4double itsWindow,
                                             const G4String& itsInputName, const bool& IsCCSorter)
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
    m_multiplesPolicy(kKeepIfAllAreGoodsOld),
    m_allPulseOpenCoincGate(false),
    m_depth(1),
    m_presortBufferSize(256),
    m_presortWarning(false),
    m_CCSorter(IsCCSorter),
    m_triggerOnlyByAbsorber(0),
    m_eventIDCoinc(0)
{
	G4cout<<"GateCoincidenceSorterOld constr"<<G4endl;

  // Create the messenger
  m_messenger = new GateCoincidenceSorterOldMessenger(this);
  //if(m_CCSorter==true)

  coincID_CC=0;
 // itsDigitizer->InsertDigiMakerModule( new GateCoincidenceDigiMaker(itsDigitizer, itsOutputName,true) );
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Destructor
GateCoincidenceSorterOld::~GateCoincidenceSorterOld()
{
  while(m_presortBuffer.size() > 0)
  {
     // G4cout<<"[GateCoincidenceSorterOld::~GateCoincidenceSorterOld()] m_presortBuffer.size="<<m_presortBuffer.size()<<G4endl;
    delete m_presortBuffer.back();
    m_presortBuffer.pop_back();
  }

  while(m_coincidencePulses.size() > 0)
  {
    delete m_coincidencePulses.back();
    m_coincidencePulses.pop_back();
  }

  delete m_messenger;
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Overload of the virtual method declared by the base class GateVCoincidenceSorter
// print-out the attributes specific of the sorter
void GateCoincidenceSorterOld::Describe(size_t indent)
{
  GateClockDependent::Describe(indent);
  G4cout << GateTools::Indent(indent) << "Coincidence window:  " << G4BestUnit(m_coincidenceWindow,"Time") << Gateendl;
  G4cout << GateTools::Indent(indent) << "Coincidence window jitter: " << G4BestUnit(m_coincidenceWindowJitter,"Time") << Gateendl;
  G4cout << GateTools::Indent(indent) << "Coincidence offset:  " << G4BestUnit(m_offset,"Time") << Gateendl;
  G4cout << GateTools::Indent(indent) << "Coincidence offset jitter: " << G4BestUnit(m_offsetJitter,"Time") << Gateendl;
  G4cout << GateTools::Indent(indent) << "Min sector diff.:    " << m_minSectorDifference << Gateendl;
  G4cout << GateTools::Indent(indent) << "Presort buffer size: " << m_presortBufferSize << Gateendl;
  G4cout << GateTools::Indent(indent) << "Input:              '" << m_inputName << "'" << Gateendl;
  G4cout << GateTools::Indent(indent) << "Output:             '" << m_outputName << "'" << Gateendl;
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
void GateCoincidenceSorterOld::SetMultiplesPolicy(const G4String& policy)
{
    if (policy=="takeWinnerOfGoodsOld")
    	m_multiplesPolicy=kTakeWinnerOfGoodsOld;
    else if (policy=="takeWinnerIfIsGoodOld")
    	m_multiplesPolicy=kTakeWinnerIfIsGoodOld;
    else if (policy=="takeWinnerIfAllAreGoodsOld")
    	m_multiplesPolicy=kTakeWinnerIfAllAreGoodsOld;
    else if (policy=="killAll")
    	m_multiplesPolicy=kKillAllOld;
    else if (policy=="takeAllGoodsOld")
    	m_multiplesPolicy=kTakeAllGoodsOld;
    else if (policy=="killAllIfMultipleGoodsOld")
    	m_multiplesPolicy=kKillAllIfMultipleGoodsOld;
    else if (policy=="keepIfAnyIsGoodOld")
    	m_multiplesPolicy=kKeepIfAnyIsGoodOld;
    else if (policy=="keepIfOnlyOneGoodOld")
    	m_multiplesPolicy=kKeepIfOnlyOneGoodOld;
    else if (policy=="keepAll")
    	m_multiplesPolicy=kKeepAllOld;
    else {
    	if (policy!="keepIfAllAreGoodsOld")
    	    G4cout<<"WARNING : policy not recognized, using default : keepMultiplesIfAllAreGoodsOld\n";
  	m_multiplesPolicy=kKeepIfAllAreGoodsOld;
    }
}
//------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------
void GateCoincidenceSorterOld::SetAcceptancePolicy4CC(const G4String &policy)
{
    if (policy=="keepAll")
        m_acceptance_policy_4CC=kKeepAllOld_CC;
    else if (policy=="keepIfMultipleVolumeIDsInvolved")
        m_acceptance_policy_4CC=kkeepIfMultipleVolumeIDsInvolvedOld_CC;
    else if (policy=="keepIfMultipleVolumeNamesInvolved")
        m_acceptance_policy_4CC=kkeepIfMultipleVolumeNamesInvolvedOld_CC;
    else {
          G4cout<<"WARNING : acceptance policy  for CC not recognized, using default : kkeepIfMultipleVolumeNamesInvolved\n";
          m_acceptance_policy_4CC=kkeepIfMultipleVolumeNamesInvolvedOld_CC;
    }
}
//------------------------------------------------------------------------------------------------------


void GateCoincidenceSorterOld::ProcessSinglePulseList(GatePulseList* inp)
{
  GatePulse* pulse;
  std::list<GatePulse*>::iterator buf_iter;                // presort buffer iterator
  std::deque<GateCoincidencePulse*>::iterator coince_iter; // coincidence list iterator

  G4bool inCoincidence;

  GateCoincidencePulse* coincidence;
  G4double window, offset;

  GatePulseIterator gpl_iter;      // input pulse list iterator

  if (!IsEnabled())
    return;

//  if(inp!=0){
//      G4cout<<"######before"<<G4endl;
//      G4cout<<"size of input pulse= "<<inp->size()<<G4endl;
//      GatePulseConstIterator iterIn;
//      for (iterIn = inp->begin() ; iterIn != inp->end() ; ++iterIn){
//          GatePulse* inp1 = *iterIn;
//          G4cout<<"evtID= "<<inp1->GetEventID()<<"  energy="<< inp1->GetEnergy()<<G4endl;
//      }
//  }



  GatePulseList* inputPulseList = inp ? inp : m_digitizer->FindPulseList( m_inputName ); //Loaded by eventID

  //if(inputPulseList!=0){
   // G4cout<<m_inputName<<G4endl;
    //G4cout<<"size of input pulse= "<<inputPulseList->size()<<G4endl;
   // GatePulseConstIterator iterIn;
  //for (iterIn = inputPulseList->begin() ; iterIn != inputPulseList->end() ; ++iterIn){
    //GatePulse* input1 = *iterIn;
    //G4cout<<"evtID= "<<input1->GetEventID()<<"  energy="<< input1->GetEnergy()<<"  time="<< input1->GetTime()<<G4endl;
  //}
  //}


  if (!inputPulseList)
    return ;

  if(m_eventIDCoinc){
      bool isCoincCreated=false;
      if(inputPulseList->size()>1){
          if(m_coincidenceWindowJitter > 0.0)
            window = G4RandGauss::shoot(m_coincidenceWindow,m_coincidenceWindowJitter);
          else
            window = m_coincidenceWindow;

          if(m_offsetJitter > 0.0)
            offset = G4RandGauss::shoot(m_offset,m_offsetJitter);
          else
            offset = m_offset;


          for(gpl_iter = inputPulseList->begin();gpl_iter != inputPulseList->end();gpl_iter++)
          {
              pulse = new GatePulse(**gpl_iter);
              if(!isCoincCreated){
                  isCoincCreated=true;
                  coincidence = new GateCoincidencePulse(m_outputName,pulse,window,offset);
              }
              else{
                   coincidence->push_back(new GatePulse(pulse)); // add a copy so we can delete safely
                   delete pulse;
              }

          }

          if(m_CCSorter==true){
              ProcessCompletedCoincidenceWindow4CC(coincidence);
          }
          else{
              ProcessCompletedCoincidenceWindow(coincidence);
          }

      }
   return;
  }







  //------ put input pulses in sorted input buffer----------
  for(gpl_iter = inputPulseList->begin();gpl_iter != inputPulseList->end();gpl_iter++)
  {
      // make a copy of the pulse
      pulse = new GatePulse(**gpl_iter);

      if(m_presortBuffer.empty())
          m_presortBuffer.push_back(pulse);
      else if(pulse->GetTime() < m_presortBuffer.back()->GetTime())    // check that even isn't earlier than the earliest event in the buffer
      {
          if(!m_presortWarning)
              GateWarning("Event is earlier than earliest event in coincidence presort buffer. Consider using a larger buffer.");
          m_presortWarning = true;
          m_presortBuffer.push_back(pulse); // this will probably not cause a problem, but coincidences may be missed
      }
      else // put the event into the presort buffer in the right place
      {
          buf_iter = m_presortBuffer.begin();
          while(pulse->GetTime() < (*buf_iter)->GetTime())
              buf_iter++;
          m_presortBuffer.insert(buf_iter, pulse);
          //      G4cout<<"presortBuffer filled in position "<<std::distance(m_presortBuffer.begin(),buf_iter)<<G4endl;
          //      G4cout<<"pulseTime "<<pulse->GetTime()<<G4endl;
          //      G4cout<<"pulseTime "<<pulse->GetTime()/ns<<G4endl;
      }

  }


  //  once buffer reaches the specified size look for coincidences
  for(G4int i = m_presortBuffer.size();i > m_presortBufferSize;i--)
  {

    pulse = m_presortBuffer.back();
    m_presortBuffer.pop_back();

    // process completed coincidence pulse window at front of list
    while(!m_coincidencePulses.empty() && m_coincidencePulses.front()->IsAfterWindow(pulse))
    {

        coincidence = m_coincidencePulses.front();

        m_coincidencePulses.pop_front();

        if(m_CCSorter==true){
            ProcessCompletedCoincidenceWindow4CC(coincidence);
        }
        else{
            ProcessCompletedCoincidenceWindow(coincidence);
        }
    }

    // add event to coincidences
    inCoincidence = false;
    coince_iter = m_coincidencePulses.begin();
    while( coince_iter != m_coincidencePulses.end() && (*coince_iter)->IsInCoincidence(pulse) )
    {
      inCoincidence = true;
       //AE here fill coincidence
      (*coince_iter)->push_back(new GatePulse(pulse)); // add a copy so we can delete safely
      coince_iter++;
    }

    // if not after or in the windows, it must be before the rest of coincidence windows
    // so there's no need to check the rest of the coincidence list

    // update coincidence pulse list
        if(m_allPulseOpenCoincGate || !inCoincidence)
        {
          if(m_coincidenceWindowJitter > 0.0)
            window = G4RandGauss::shoot(m_coincidenceWindow,m_coincidenceWindowJitter);
          else
            window = m_coincidenceWindow;

          if(m_offsetJitter > 0.0)
            offset = G4RandGauss::shoot(m_offset,m_offsetJitter);
          else
            offset = m_offset;

          if(m_triggerOnlyByAbsorber==1){

              if(((pulse->GetVolumeID()).GetBottomCreator())->GetObjectName()==m_absorberSD){
              //if(pulse->GetVolumeID().GetVolume(2)->GetName()==m_absorberDepth2Name){
                  coincidence = new GateCoincidencePulse(m_outputName,pulse,window,offset);
                   //AE here open coincidence
                   m_coincidencePulses.push_back(coincidence);

              }
          }
          else{
            coincidence = new GateCoincidencePulse(m_outputName,pulse,window,offset);
             //AE here open window with the pulse
             m_coincidencePulses.push_back(coincidence);
          }
        }
        else
          delete pulse; // pulses that don't open a coincidence window can be discarded
  }

}


void GateCoincidenceSorterOld::ProcessCompletedCoincidenceWindow4CC(GateCoincidencePulse *coincidence)
{


    G4int nPulses = coincidence->size();
    if (nPulses<2)
    {
        delete coincidence;
        return;
    }
    else if (nPulses>=2)
    {

        if(IsCoincidenceGood4CC(coincidence)==true){
            // Introduce some conditions to check if  is good
            coincidence->SetCoincID(coincID_CC);
            m_digitizer->StoreCoincidencePulse(coincidence);
            coincID_CC++;
            return;
        }
       else{
             //G4cout<<"bad coinc"<<G4endl;
            delete coincidence;
            return;

        }
    }

    delete coincidence;
    return;
}

// look for valid coincidences
void GateCoincidenceSorterOld::ProcessCompletedCoincidenceWindow(GateCoincidencePulse *coincidence)
{
  G4int i, j, nPulses;
  G4int nGoodsOld, maxGoodsOld;
  G4double E, maxE;
  G4int winner_i=0;
  G4int winner_j=1;

  G4bool PairWithFirstPulseOnly;

  nPulses = coincidence->size();

  if (nPulses<2)
  {
    delete coincidence;
    return;
  }
  else if (nPulses==2)
  {
    // check if good
    if(IsForbiddenCoincidence(coincidence->at(0),coincidence->at(1)) )
      delete coincidence;
    else
      m_digitizer->StoreCoincidencePulse(coincidence);
    return;
  }
  else // nPulses>2 multiples
  {
    if(m_multiplesPolicy==kKillAllOld)
    {
      delete coincidence;
      return;
    }

    // if dealing with a delayed window or if other pulses open coincidence windows,
    // we only want to pair with the first pulse to avoid invalid pairs, or double counting
    PairWithFirstPulseOnly = m_allPulseOpenCoincGate | coincidence->IsDelayed();

    if(m_multiplesPolicy==kTakeAllGoodsOld)
    {
      for(i=0; i<(PairWithFirstPulseOnly?1:(nPulses-1)); i++) // iterate over all pairs (single window) or just pairs with initial event (multi-window)
        for(j=i+1; j<nPulses; j++)
          if(!IsForbiddenCoincidence(coincidence->at(i),coincidence->at(j)) )
            m_digitizer->StoreCoincidencePulse(CreateSubPulse(coincidence, i, j));
      delete coincidence; // valid pulses extracted so we can delete
      return;
    }

    // count the goods (iterate over all pairs because we're considering the multi as a unit, not breaking it up into pairs)
    nGoodsOld = 0;
    for(i=0; i<(coincidence->IsDelayed()?1:(nPulses-1)); i++)
      for(j=i+1; j<nPulses; j++)
        if(!IsForbiddenCoincidence(coincidence->at(i),coincidence->at(j)))
          nGoodsOld++;

    if( nGoodsOld == 0 )  // all of the remaining options expect at least one good
    {
      delete coincidence;
      return;
    }

    // all the Keep* policies pass on a multi-coincidence rather than breaking into pairs
    if( ( (m_multiplesPolicy==kKeepIfAnyIsGoodOld) /*&& (nGoodsOld>0)*/          ) || // if nGoodsOld = 0, we don't get here
        ( (m_multiplesPolicy==kKeepIfOnlyOneGoodOld) && (nGoodsOld==1)           ) ||
        ( (m_multiplesPolicy==kKeepIfAllAreGoodsOld) && (nGoodsOld==(nPulses*(nPulses-1)/2)) ) )
    {
      m_digitizer->StoreCoincidencePulse(coincidence);
      return; // don't delete the coincidence
    }
    if((m_multiplesPolicy==kKeepIfAnyIsGoodOld)   ||
       (m_multiplesPolicy==kKeepIfOnlyOneGoodOld) ||
       (m_multiplesPolicy==kKeepIfAllAreGoodsOld) )
    {
      delete coincidence;
      return;
    }

    // find winner and count the goods
    maxE = 0.0;
    nGoodsOld = 0;
    for(i=0; i<(PairWithFirstPulseOnly?1:(nPulses-1)); i++)
      for(j=i+1; j<nPulses; j++)
      {
        // this time we might only be counting goods on the subset involving the first event
        if(!IsForbiddenCoincidence(coincidence->at(i),coincidence->at(j)))
          nGoodsOld++;

        E = coincidence->at(i)->GetEnergy() + coincidence->at(j)->GetEnergy();
        if(E>maxE)
        {
          maxE = E;
          winner_i = i;
          winner_j = j;
        }
      }

    if(nGoodsOld==0) // check again, we may have reduced the subset.
    {
      delete coincidence;
      return;
    }

    if(m_multiplesPolicy==kTakeWinnerIfIsGoodOld)
    {
      if(!IsForbiddenCoincidence(coincidence->at(winner_i),coincidence->at(winner_j)) )
        m_digitizer->StoreCoincidencePulse(CreateSubPulse(coincidence, winner_i, winner_j));
      delete coincidence;
      return;
    }

    if(m_multiplesPolicy==kKillAllIfMultipleGoodsOld)
    {
      if(nGoodsOld>1)
      {
        delete coincidence;
        return;
      } // else find and return the one good event
      else // nGoodsOld==1
      {
        for(i=0; i<(coincidence->IsDelayed()?1:(nPulses-1)); i++)
          for(j=i+1; j<nPulses; j++)
            if(!IsForbiddenCoincidence(coincidence->at(i),coincidence->at(j)))
              m_digitizer->StoreCoincidencePulse(CreateSubPulse(coincidence, i, j));
        delete coincidence;
        return;
      }
    }

    maxGoodsOld = PairWithFirstPulseOnly?(nPulses-1):(nPulses*(nPulses-1)/2);
    if(m_multiplesPolicy==kTakeWinnerIfAllAreGoodsOld)
    {
      if(nGoodsOld==maxGoodsOld)
      {
        m_digitizer->StoreCoincidencePulse(CreateSubPulse(coincidence, winner_i, winner_j));
        delete coincidence;
        return;
      }
      else
      {
        delete coincidence;
        return;
      }
    }

    if(m_multiplesPolicy==kTakeWinnerOfGoodsOld)
    {
      // find winner
      maxE = 0.0;
      for(i=0; i<(PairWithFirstPulseOnly?1:(nPulses-1)); i++)
        for(j=i+1; j<nPulses; j++)
        {
          if(!IsForbiddenCoincidence(coincidence->at(i),coincidence->at(j)))
          {
            E = coincidence->at(i)->GetEnergy() + coincidence->at(j)->GetEnergy();
            if(E>maxE)
            {
              maxE = E;
              winner_i = i;
              winner_j = j;
            }
          }
        }
      m_digitizer->StoreCoincidencePulse(CreateSubPulse(coincidence, winner_i, winner_j));
      delete coincidence; // valid pulses extracted so we can delete
      return;
    }
  }

  delete coincidence;
  return;

}

GateCoincidencePulse* GateCoincidenceSorterOld::CreateSubPulse(GateCoincidencePulse* coincidence, G4int i, G4int j)
{
  GatePulse* pulse1 = new GatePulse(coincidence->at(i));
  GatePulse* pulse2 = new GatePulse(coincidence->at(j));
  G4double offset = coincidence->GetStartTime() - pulse1->GetTime();
  GateCoincidencePulse *newCoincPulse = new GateCoincidencePulse(m_outputName,pulse1,m_coincidenceWindow,offset);
  newCoincPulse->push_back(pulse2);
  return newCoincPulse;
}

//------------------------------------------------------------------------------------------------------
G4int GateCoincidenceSorterOld::ComputeSectorID(const GatePulse& pulse)
{
    if (m_depth>=(G4int)pulse.GetOutputVolumeID().size()) {
    	G4cerr<<"[GateCoincidenceSorterOld::ComputeSectorID]: Required depth's too deep, setting it to 1\n";
	m_depth=1;
    }
    static std::vector<G4int> gkSectorMultiplier;
    static std::vector<G4int> gkSectorNumber;
    if (gkSectorMultiplier.empty()){
    	// this code is done just one time for performance improving
	// one suppose that the system hierarchy is linear until the desired depth
    	GateSystemComponent* comp = m_system->GetBaseComponent();
	G4int depth=0;
	while (comp){
	    G4int rep_num = comp->GetAngularRepeatNumber();
	    if (rep_num == 1)
	        // Check for generic repeater
	        rep_num = comp->GetGenericRepeatNumber();
	    gkSectorNumber.push_back(rep_num);

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
  G4bool GateCoincidenceSorterOld::IsCoincidenceGood4CC(GateCoincidencePulse *coincidence){
      G4bool isGoodOld=false;
      if(m_acceptance_policy_4CC==kKeepAllOld_CC){
          return true;
      }
      else if (m_acceptance_policy_4CC==kkeepIfMultipleVolumeIDsInvolvedOld_CC){
          GateVolumeID volID1=coincidence->at(0)->GetVolumeID();
          G4String volBName1=((coincidence->at(0)->GetVolumeID()).GetBottomCreator())->GetObjectName();
          G4int cpN1=volID1.GetBottomVolume()->GetCopyNo();
          volBName1=volBName1+std::to_string(cpN1);
          //Restriction at least coincidence in two different volumesID

          unsigned int numCoincPulses=coincidence->size();
         for(unsigned int i=1;i<numCoincPulses;i++){
                G4String volN_i=((coincidence->at(i)->GetVolumeID()).GetBottomCreator())->GetObjectName();
                G4int cpN_i=(coincidence->at(i)->GetVolumeID()).GetBottomVolume()->GetCopyNo();
                //Distinguish between sensitive volumes (treating repeaters as different volumes)
                if((volN_i+std::to_string(cpN_i))!=volBName1){
                    //G4cout<<(volN_i+std::to_string(cpN_i))<<G4endl;
                   //G4cout<< volBName1<<G4endl;
                    isGoodOld=true;
                    break;
                }

          }

      }
      else if(m_acceptance_policy_4CC==kkeepIfMultipleVolumeNamesInvolvedOld_CC){
          //GateVolumeID volID1=coincidence->at(0)->GetVolumeID();
          G4String volBName1=((coincidence->at(0)->GetVolumeID()).GetBottomCreator())->GetObjectName();
          unsigned int numCoincPulses=coincidence->size();

          for(unsigned int i=1;i<numCoincPulses;i++){
                //it= find (diffVID.begin(),diffVID.end(), coincidence->at(i)->GetVolumeID());
                //G4String volN_i=((coincidence->at(i)->GetVolumeID()).GetBottomCreator())->GetObjectName();
                if(((coincidence->at(i)->GetVolumeID()).GetBottomCreator())->GetObjectName()!=volBName1){
                    isGoodOld=true;
                    break;
                }
          }

      }
      else{
          G4cout<<"[GateCoincidenceSorterOld]: Problems in CC accpetance policy"<<G4endl;
      }

    return isGoodOld;
  }

//------------------------------------------------------------------------------------------------------
// Check whether a coincidence is invalid: ring difference or sector difference too small...
G4bool GateCoincidenceSorterOld::IsForbiddenCoincidence(const GatePulse* pulse1, const GatePulse* pulse2)
{
  G4int blockID1 = m_system->GetMainComponentID(pulse1),
        blockID2 = m_system->GetMainComponentID(pulse2);

 // Modif by D. Lazaro, February 25th, 2004
  // Computation of sectorID, sectorNumber and sectorDifference, paramaters depending on
  // the geometry construction of the scanner (spherical for system ecatAccel and cylindrical
  // for other systems as Ecat, CPET and cylindricalPET)

  const G4String name = m_system->GetName();
  G4String nameComp = "systems/ecatAccel";
  //G4cout << "NAME OF THE SYSTEM: " << name << "; NAME TO COMPARE: " << nameComp << Gateendl;
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
        G4cout << "[GateCoincidenceSorterOld::IsForbiddenCoincidence]: coincidence between neighbor blocks --> refused\n";
      return true;
    }
    return false;
  }
  else {
  // Compute the sector difference
  G4int sectorID1 = ComputeSectorID(*pulse1),
      	sectorID2 = ComputeSectorID(*pulse2);

  // Get the number of sectors per ring
  // G4int sectorNumber = GetCoincidentSectorNumber();
  // Deal with the circular difference problem
  G4int sectorDiff1 = sectorID1 - sectorID2;
  if (sectorDiff1<0)
    sectorDiff1 += gm_coincSectNum;
  G4int sectorDiff2 = sectorID2 - sectorID1;
  if (sectorDiff2<0)
    sectorDiff2 += gm_coincSectNum;
  G4int sectorDifference = std::min(sectorDiff1,sectorDiff2);

  //Compare the sector difference with the minimum differences for valid coincidences
  if (sectorDifference<m_minSectorDifference) {
      	if (nVerboseLevel>1)
      	    G4cout << "[GateCoincidenceSorterOld::IsForbiddenCoincidence]: coincidence between neighbour blocks --> refused\n";
	return true;
  }

  return false;
  }
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Next method was added for the multi-system approach

void GateCoincidenceSorterOld::SetSystem(G4String& inputName)
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
