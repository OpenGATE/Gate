/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateCoincidenceSorter.hh"

#include "Randomize.hh"

#include "G4UnitsTable.hh"

#include "GateVolumeID.hh"
#include "GateObjectStore.hh"


#include "GateCoincidenceSorterMessenger.hh"

#include "GateTools.hh"
#include "GateDigitizerMgr.hh"
#include "GateVSystem.hh"
#include "GateOutputMgr.hh"


//------------------------------------------------------------------------------------------------------
G4int GateCoincidenceSorter::gm_coincSectNum=0;
// Constructs a new coincidence sorter, attached to a GateDigitizerOld and to a system
GateCoincidenceSorter::GateCoincidenceSorter(GateDigitizerMgr* itsDigitizerMgr,
                                             const G4String& itsOutputName,
                                             const bool& IsCCSorter)
 	: GateVDigitizerModule("GateCoincidenceSorter", "digitizerMgr/CoincidenceSorter/"+itsOutputName),
    m_digitizerMgr(itsDigitizerMgr),
    m_system(0),
    m_outputName(itsOutputName),
    m_coincidenceWindow(10.* ns),
    m_coincidenceWindowJitter(0.),
    m_offset(0.),
    m_offsetJitter(0.),
    m_minSectorDifference(2),
    m_multiplesPolicy(kKeepIfAllAreGoods),
    m_allDigiOpenCoincGate(false),
    m_depth(1),
    m_presortBufferSize(256),
    m_presortWarning(false),
    m_CCSorter(IsCCSorter),
    m_triggerOnlyByAbsorber(0),
    m_eventIDCoinc(0)
{

  // Create the messenger
  m_messenger = new GateCoincidenceSorterMessenger(this);
  //if(m_CCSorter==true)

  coincID_CC=0;
  //OK GND 2022
  GateOutputMgr::GetInstance()->RegisterNewCoincidenceDigiCollection(m_outputName,true);

  G4String colName = itsOutputName;
  collectionName.push_back(colName);


  G4DigiManager::GetDMpointer()->AddNewModule(this);



}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Destructor
GateCoincidenceSorter::~GateCoincidenceSorter()
{
  while(m_presortBuffer.size() > 0)
  {
    delete m_presortBuffer.back();
    m_presortBuffer.pop_back();
  }

  while(m_coincidenceDigis.size() > 0)
  {
    delete m_coincidenceDigis.back();
    m_coincidenceDigis.pop_back();
  }

  delete m_messenger;
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
// Overload of the virtual method declared by the base class GateVCoincidenceSorter
// print-out the attributes specific of the sorter
void GateCoincidenceSorter::DescribeMyself(size_t indent)
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

void GateCoincidenceSorter::Describe(size_t indent)
{
	DescribeMyself(indent);
}

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
    	    G4cout<<"WARNING : policy not recognized, using default : keepMultiplesIfAllAreGoods\n";
  	m_multiplesPolicy=kKeepIfAllAreGoods;
    }
}
//------------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------
void GateCoincidenceSorter::SetAcceptancePolicy4CC(const G4String &policy)
{
    if (policy=="keepAll")
        m_acceptance_policy_4CC=kKeepAll_CC;
    else if (policy=="keepIfMultipleVolumeIDsInvolved")
        m_acceptance_policy_4CC=kkeepIfMultipleVolumeIDsInvolved_CC;
    else if (policy=="keepIfMultipleVolumeNamesInvolved")
        m_acceptance_policy_4CC=kkeepIfMultipleVolumeNamesInvolved_CC;
    else {
          G4cout<<"WARNING : acceptance policy  for CC not recognized, using default : kkeepIfMultipleVolumeNamesInvolved\n";
          m_acceptance_policy_4CC=kkeepIfMultipleVolumeNamesInvolved_CC;
    }
}
//------------------------------------------------------------------------------------------------------


void GateCoincidenceSorter::Digitize()
{
	//G4cout<<"GateCoincidenceSorter::Digitize "<< GetOutputName() <<G4endl;
		//G4cout<< "m_inputName "<< m_inputName<<G4endl;
  GateDigi* digi;
  std::list<GateDigi*>::iterator buf_iter;                // presort buffer iterator

  std::deque<GateCoincidenceDigi*>::iterator coince_iter; // coincidence list iterator

  G4bool inCoincidence;

  GateCoincidenceDigi* coincidence;
  G4double window, offset;

  //Input digi collection
  GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();
  GateSinglesDigitizer* inputDigitizer;
  inputDigitizer = digitizerMgr->FindSinglesDigitizer(m_inputName);//m_collectionName);
  if (!inputDigitizer)
	  if (digitizerMgr->m_SDlist.size()==1)
  	  {
		  G4String new_name= m_inputName+"_"+digitizerMgr->m_SDlist[0]->GetName();
		  //G4cout<<" new_name "<< new_name<<G4endl;
		  inputDigitizer = digitizerMgr->FindSinglesDigitizer(new_name);
  	  }
	  else
		  GateError("ERROR: The name _"+ m_inputName+"_ is unknown for input singles digicollection! \n");

  G4int inputCollID=inputDigitizer->m_outputDigiCollectionID;
  //G4cout<<"inputCollID "<<inputCollID<<G4endl;
  G4DigiManager *fDM = G4DigiManager::GetDMpointer();

  GateDigiCollection* IDC = 0;
  IDC = (GateDigiCollection*) (fDM->GetDigiCollection(inputCollID));
  if (!IDC)
     return ;

  std::vector< GateDigi* >* IDCvector = IDC->GetVector ();
  std::vector<GateDigi*>::iterator gpl_iter;

  //Output digi collection
  m_OutputCoincidenceDigiCollection = new GateCoincidenceDigiCollection("GateCoincidenceSorter",m_outputName); // to create the Digi Collection


  if (!IsEnabled())
     return;



  if(m_eventIDCoinc){
      bool isCoincCreated=false;
      if(IDCvector->size()>1){
          if(m_coincidenceWindowJitter > 0.0)
            window = G4RandGauss::shoot(m_coincidenceWindow,m_coincidenceWindowJitter);
          else
            window = m_coincidenceWindow;

          if(m_offsetJitter > 0.0)
            offset = G4RandGauss::shoot(m_offset,m_offsetJitter);
          else
            offset = m_offset;


          for(gpl_iter = IDCvector->begin();gpl_iter != IDCvector->end();gpl_iter++)
          {
              digi = new GateDigi(**gpl_iter);
              if(!isCoincCreated){
                  isCoincCreated=true;
                  coincidence = new GateCoincidenceDigi(digi,window,offset);
              }
              else{
                   coincidence->push_back(new GateDigi(digi)); // add a copy so we can delete safely
                   delete digi;
              }

          }



          if(m_CCSorter==true){
           //TODO: CC coincidences
            //  ProcessCompletedCoincidenceWindow4CC(coincidence);
          }
          else{
              ProcessCompletedCoincidenceWindow(coincidence);
          }

      }
   return;
  }



  //------ put input digis in sorted input buffer----------
  for(gpl_iter = IDCvector->begin();gpl_iter != IDCvector->end();gpl_iter++)
  {
      // make a copy of the digi
      digi = new GateDigi(**gpl_iter);

      if(m_presortBuffer.empty())
    	  m_presortBuffer.push_back(digi);
      else if(digi->GetTime() < m_presortBuffer.back()->GetTime())    // check that even isn't earlier than the earliest event in the buffer
      {
          if(!m_presortWarning)
              GateWarning("Event is earlier than earliest event in coincidence presort buffer. Consider using a larger buffer (/setPresortBufferSize n, where n>256)");
          m_presortWarning = true;
          m_presortBuffer.push_back(digi); // this will probably not cause a problem, but coincidences may be missed
      }
      else // put the event into the presort buffer in the right place
      {
          buf_iter = m_presortBuffer.begin();
          while(digi->GetTime() < (*buf_iter)->GetTime())
              buf_iter++;
          m_presortBuffer.insert(buf_iter, digi);
          // G4cout<<"presortBuffer filled in position "<<std::distance(m_presortBuffer.begin(),buf_iter)<<G4endl;
          // G4cout<<"digiTime "<<digi->GetTime()<<G4endl;
          // G4cout<<"digiTime "<<digi->GetTime()/ns<<G4endl;
      }

  }


  //  once buffer reaches the specified size look for coincidences
  for(G4int i = m_presortBuffer.size();i > m_presortBufferSize;i--)
  {
    digi = m_presortBuffer.back();
    m_presortBuffer.pop_back();
    // process completed coincidence pulse window at front of list
    while(!m_coincidenceDigis.empty() && m_coincidenceDigis.front()->IsAfterWindow(digi))
    {
    	coincidence = m_coincidenceDigis.front();

    	m_coincidenceDigis.pop_front();

        if(m_CCSorter==true){
        //TODO CC sorter
           // ProcessCompletedCoincidenceWindow4CC(coincidence);
        }
        else{
            ProcessCompletedCoincidenceWindow(coincidence);
        }

   }
    // add event to coincidences
    inCoincidence = false;
    coince_iter = m_coincidenceDigis.begin();
    while( coince_iter != m_coincidenceDigis.end() && (*coince_iter)->IsInCoincidence(digi) )
    {
      inCoincidence = true;
       //AE here fill coincidence
      (*coince_iter)->push_back(new GateDigi(digi)); // add a copy so we can delete safely
      coince_iter++;
    }

    // if not after or in the windows, it must be before the rest of coincidence windows
    // so there's no need to check the rest of the coincidence list

    // update coincidence digi list
        if(m_allDigiOpenCoincGate || !inCoincidence)
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

              if(((digi->GetVolumeID()).GetBottomCreator())->GetObjectName()==m_absorberSD){
              //if(digi->GetVolumeID().GetVolume(2)->GetName()==m_absorberDepth2Name){
                  coincidence = new GateCoincidenceDigi(digi,window,offset);
                   //AE here open coincidence
                   m_coincidenceDigis.push_back(coincidence);

              }
          }
          else{
            coincidence = new GateCoincidenceDigi(digi,window,offset);
             //AE here open window with the digi
             m_coincidenceDigis.push_back(coincidence);
          }
        }
        else
          delete digi; // digis that don't open a coincidence window can be discarded
  }

  StoreDigiCollection(m_OutputCoincidenceDigiCollection);




}

/*
void GateCoincidenceSorter::ProcessCompletedCoincidenceWindow4CC(GateCoincidenceDigi *coincidence)
{


    G4int nDigis = coincidence->size();
    if (nDigis<2)
    {
        delete coincidence;
        return;
    }
    else if (nDigis>=2)
    {

        if(IsCoincidenceGood4CC(coincidence)==true){
            // Introduce some conditions to check if  is good
            coincidence->SetCoincID(coincID_CC);
            m_digitizerMgr->StoreCoincidenceDigi(coincidence);
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
*/
// look for valid coincidences
void GateCoincidenceSorter::ProcessCompletedCoincidenceWindow(GateCoincidenceDigi *coincidence)
{
  G4int i, j, nDigis;
  G4int nGoods, maxGoods;
  G4double E, maxE;
  G4int winner_i=0;
  G4int winner_j=1;

  G4bool PairWithFirstDigiOnly;

  nDigis = coincidence->size();
  if (nDigis<2)
  {
    delete coincidence;
    return;
  }
  else if (nDigis==2)
  {
    // check if good
    if(IsForbiddenCoincidence(coincidence->at(0),coincidence->at(1)) )
      delete coincidence;
    else
    	m_OutputCoincidenceDigiCollection->insert(coincidence);
    return;
  }
  else // nDigis>2 multiples
  {
    if(m_multiplesPolicy==kKillAll)
    {
      delete coincidence;
      return;
    }
    // if dealing with a delayed window or if other digis open coincidence windows,
    // we only want to pair with the first digi to avoid invalid pairs, or double counting
    PairWithFirstDigiOnly = m_allDigiOpenCoincGate | coincidence->IsDelayed();

    if(m_multiplesPolicy==kTakeAllGoods)
    {
      for(i=0; i<(PairWithFirstDigiOnly?1:(nDigis-1)); i++) // iterate over all pairs (single window) or just pairs with initial event (multi-window)
        for(j=i+1; j<nDigis; j++)
          if(!IsForbiddenCoincidence(coincidence->at(i),coincidence->at(j)) )
        	  m_OutputCoincidenceDigiCollection->insert(CreateSubDigi(coincidence, i, j));
      delete coincidence; // valid digis extracted so we can delete
      return;
    }
    // count the goods (iterate over all pairs because we're considering the multi as a unit, not breaking it up into pairs)
    nGoods = 0;
    for(i=0; i<(coincidence->IsDelayed()?1:(nDigis-1)); i++)
      for(j=i+1; j<nDigis; j++)
        if(!IsForbiddenCoincidence(coincidence->at(i),coincidence->at(j)))
          nGoods++;

    if( nGoods == 0 )  // all of the remaining options expect at least one good
    {
      delete coincidence;
      return;
    }

    //G4cout<<"nGoods = "<<  nGoods<<G4endl;
    // all the Keep* policies pass on a multi-coincidence rather than breaking into pairs
    if( ( (m_multiplesPolicy==kKeepIfAnyIsGood) /*&& (nGoods>0)*/ ) || // if nGoods = 0, we don't get here
    	( (m_multiplesPolicy==kKeepIfOnlyOneGood) && (nGoods==1)           ) ||
        ( (m_multiplesPolicy==kKeepIfAllAreGoods) && (nGoods==(nDigis*(nDigis-1)/2)) ) )
    {
    	m_OutputCoincidenceDigiCollection->insert(coincidence);
      return; // don't delete the coincidence
    }
    if((m_multiplesPolicy==kKeepIfAnyIsGood)   ||
       (m_multiplesPolicy==kKeepIfOnlyOneGood) ||
       (m_multiplesPolicy==kKeepIfAllAreGoods) )
    {
      delete coincidence;
      return;
    }
    // find winner and count the goods
    maxE = 0.0;
    nGoods = 0;
    for(i=0; i<(PairWithFirstDigiOnly?1:(nDigis-1)); i++)
      for(j=i+1; j<nDigis; j++)
      {
        // this time we might only be counting goods on the subset involving the first event
        if(!IsForbiddenCoincidence(coincidence->at(i),coincidence->at(j)))
          nGoods++;

        E = coincidence->at(i)->GetEnergy() + coincidence->at(j)->GetEnergy();
        if(E>maxE)
        {
          maxE = E;
          winner_i = i;
          winner_j = j;
        }
      }
    if(nGoods==0) // check again, we may have reduced the subset.
    {
      delete coincidence;
      return;
    }

    if(m_multiplesPolicy==kTakeWinnerIfIsGood)
    {
      if(!IsForbiddenCoincidence(coincidence->at(winner_i),coincidence->at(winner_j)) )
    	  m_OutputCoincidenceDigiCollection->insert(CreateSubDigi(coincidence, winner_i, winner_j));
      delete coincidence;
      return;
    }
    if(m_multiplesPolicy==kKillAllIfMultipleGoods)
    {
      if(nGoods>1)
      {
        delete coincidence;
        return;
      } // else find and return the one good event
      else // nGoods==1
      {
        for(i=0; i<(coincidence->IsDelayed()?1:(nDigis-1)); i++)
          for(j=i+1; j<nDigis; j++)
            if(!IsForbiddenCoincidence(coincidence->at(i),coincidence->at(j)))
            	m_OutputCoincidenceDigiCollection->insert(CreateSubDigi(coincidence, i, j));
        delete coincidence;
        return;
      }
    }
    maxGoods = PairWithFirstDigiOnly?(nDigis-1):(nDigis*(nDigis-1)/2);
    if(m_multiplesPolicy==kTakeWinnerIfAllAreGoods)
    {
      if(nGoods==maxGoods)
      {
    	  m_OutputCoincidenceDigiCollection->insert(CreateSubDigi(coincidence, winner_i, winner_j));
        delete coincidence;
        return;
      }
      else
      {
        delete coincidence;
        return;
      }
    }

    if(m_multiplesPolicy==kTakeWinnerOfGoods)
    {
      // find winner
      maxE = 0.0;
      for(i=0; i<(PairWithFirstDigiOnly?1:(nDigis-1)); i++)
        for(j=i+1; j<nDigis; j++)
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
	  m_OutputCoincidenceDigiCollection->insert(CreateSubDigi(coincidence, winner_i, winner_j));
      delete coincidence; // valid digis extracted so we can delete
      return;
    }
  }
  delete coincidence;
  return;

}

GateCoincidenceDigi* GateCoincidenceSorter::CreateSubDigi(GateCoincidenceDigi* coincidence, G4int i, G4int j)
{
  GateDigi* digi1 = new GateDigi(coincidence->at(i));
  GateDigi* digi2 = new GateDigi(coincidence->at(j));
  G4double offset = coincidence->GetStartTime() - digi1->GetTime();
  GateCoincidenceDigi *newCoincDigi = new GateCoincidenceDigi(digi1,m_coincidenceWindow,offset);
  newCoincDigi->push_back(digi2);
  return newCoincDigi;
}

//------------------------------------------------------------------------------------------------------
G4int GateCoincidenceSorter::ComputeSectorID(const GateDigi& digi)
{
    if (m_depth>=(G4int)digi.GetOutputVolumeID().size()) {
    	G4cerr<<"[GateCoincidenceSorter::ComputeSectorID]: Required depth's too deep, setting it to 1\n";
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
    	G4int x = digi.GetComponentID(i)%gkSectorNumber[i];
    	ans += x*gkSectorMultiplier[i];
    }

    return ans;
}

/*
//------------------------------------------------------------------------------------------------------
  G4bool GateCoincidenceSorter::IsCoincidenceGood4CC(GateCoincidenceDigi *coincidence){
      G4bool isGood=false;
      if(m_acceptance_policy_4CC==kKeepAll_CC){
          return true;
      }
      else if (m_acceptance_policy_4CC==kkeepIfMultipleVolumeIDsInvolved_CC){
          GateVolumeID volID1=coincidence->at(0)->GetVolumeID();
          G4String volBName1=((coincidence->at(0)->GetVolumeID()).GetBottomCreator())->GetObjectName();
          G4int cpN1=volID1.GetBottomVolume()->GetCopyNo();
          volBName1=volBName1+std::to_string(cpN1);
          //Restriction at least coincidence in two different volumesID

          unsigned int numCoincDigis=coincidence->size();
         for(unsigned int i=1;i<numCoincDigis;i++){
                G4String volN_i=((coincidence->at(i)->GetVolumeID()).GetBottomCreator())->GetObjectName();
                G4int cpN_i=(coincidence->at(i)->GetVolumeID()).GetBottomVolume()->GetCopyNo();
                //Distinguish between sensitive volumes (treating repeaters as different volumes)
                if((volN_i+std::to_string(cpN_i))!=volBName1){
                    //G4cout<<(volN_i+std::to_string(cpN_i))<<G4endl;
                   //G4cout<< volBName1<<G4endl;
                    isGood=true;
                    break;
                }

          }

      }
      else if(m_acceptance_policy_4CC==kkeepIfMultipleVolumeNamesInvolved_CC){
          //GateVolumeID volID1=coincidence->at(0)->GetVolumeID();
          G4String volBName1=((coincidence->at(0)->GetVolumeID()).GetBottomCreator())->GetObjectName();
          unsigned int numCoincDigis=coincidence->size();

          for(unsigned int i=1;i<numCoincDigis;i++){
                //it= find (diffVID.begin(),diffVID.end(), coincidence->at(i)->GetVolumeID());
                //G4String volN_i=((coincidence->at(i)->GetVolumeID()).GetBottomCreator())->GetObjectName();
                if(((coincidence->at(i)->GetVolumeID()).GetBottomCreator())->GetObjectName()!=volBName1){
                    isGood=true;
                    break;
                }
          }

      }
      else{
          G4cout<<"[GateCoincidenceSorter]: Problems in CC accpetance policy"<<G4endl;
      }

    return isGood;
  }
*/
//------------------------------------------------------------------------------------------------------
// Check whether a coincidence is invalid: ring difference or sector difference too small...
G4bool GateCoincidenceSorter::IsForbiddenCoincidence(const GateDigi* digi1, const GateDigi* digi2)
{
		G4int blockID1 = m_system->GetMainComponentIDGND(digi1),
        blockID2 = m_system->GetMainComponentIDGND(digi2);

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
        G4cout << "[GateCoincidenceSorter::IsForbiddenCoincidence]: coincidence between neighbor blocks --> refused\n";
      return true;
    }
    return false;
  }
  else {
  // Compute the sector difference
  G4int sectorID1 = ComputeSectorID(*digi1),
      	sectorID2 = ComputeSectorID(*digi2);

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
   for (size_t i=0; i<m_digitizerMgr->m_SingleDigitizersList.size() ; ++i)
   {
      G4String pPCOutputName = m_digitizerMgr->m_SingleDigitizersList[i]->GetOutputName();
      if(pPCOutputName.compare(inputName) == 0)
      {
    	  m_system=m_digitizerMgr->m_SingleDigitizersList[i]->GetSystem();
         break;
      }
   }

}


