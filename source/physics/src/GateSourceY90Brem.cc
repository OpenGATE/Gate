#include "GateSourceY90Brem.hh"
#include "G4PrimaryVertex.hh"

#include <fstream>

#include "GateSourceY90BremTables.cc"

GateSourceY90Brem::GateSourceY90Brem(G4String name) : GateVSource( name )
{
  GateMessage("Beam", 1, "GateSourceY90Brem::GateSourceY90Brem(G4String) called." << Gateendl);

  m_sourceMessenger = new GateSourceY90BremMessenger(this);

  int i, j;
  ifstream input_file;

  // calculate total probability of bremsstrahlung emission for a single beta
  mBremProb = 0;
  for(i=0;i<200;i++)
    mBremProb += mEnergyTable[i];

  mCumulativeEnergyTable = new G4double[200];
  mCumulativeEnergyTable[0]  = mEnergyTable[0];
  for(i=1;i<200;i++)
    mCumulativeEnergyTable[i]=mCumulativeEnergyTable[i-1] + mEnergyTable[i];
  for(i=0;i<200;i++)
    mCumulativeEnergyTable[i] /= mCumulativeEnergyTable[199];

  G4cout << "Energy file loaded. Total brem likelihood: " << mBremProb << G4endl;

  // convert to cumulative distribution and normalize each row
  mCumulativeRangeTable = new G4double*[100];
  for(i=0;i<100;i++)
  {
    mCumulativeRangeTable[i]=new G4double[120];
    mCumulativeRangeTable[i][0] = mRangeTable[i][0];
    for(j=1;j<120;j++)
      mCumulativeRangeTable[i][j] = mCumulativeRangeTable[i][j-1] + mRangeTable[i][j];
    for(j=0;j<120;j++)
      mCumulativeRangeTable[i][j] /= mCumulativeRangeTable[i][119];
  }

  // convert to cumulative distribution and normalize each row
  mCumulativeAngleTable = new G4double*[100];
  for(i=0;i<100;i++)
  {
    mCumulativeAngleTable[i] = new G4double[180];
    mCumulativeAngleTable[i][0] = mAngleTable[i][0];
    for(j=1;j<180;j++)
      mCumulativeAngleTable[i][j] = mCumulativeAngleTable[i][j-1] + mAngleTable[i][j];
    for(j=0;j<180;j++)
      mCumulativeAngleTable[i][j] /= mCumulativeAngleTable[i][179];
  }

  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  pParticleDefinition = particleTable->FindParticle("gamma");

  m_angSPS->SetAngDistType("iso");
  mMinEnergy = 0.0;

}

GateSourceY90Brem::~GateSourceY90Brem()
{
  // delete the allocated arrays for the range and angle tables.
  int i;

  if(mCumulativeEnergyTable)
    delete [] mCumulativeEnergyTable;

  if(mCumulativeRangeTable)
  {
    for(i=0;i<100;i++)
      delete [] mCumulativeRangeTable[i];
    delete [] mCumulativeRangeTable;
  }

  if(mCumulativeAngleTable)
  {
    for(i=0;i<100;i++)
      delete [] mCumulativeAngleTable[i];
    delete [] mCumulativeAngleTable;
  }
}

G4int GateSourceY90Brem::GeneratePrimaries(G4Event *event)
{

  G4int numVertices = 0;

  G4double energy;
  G4ThreeVector position;
  G4ThreeVector direction;      // direction vector from beta particle source to point of brem
  G4ThreeVector momentum_dir;

  GateMessage("Beam", 1, "GateSourceY90Brem::GeneratePrimaries(G4Event*) called at " << m_time << " s." << Gateendl);
  SetParticleTime(m_time);

  // sample energy distribution
  energy = GetEnergy();
  if(energy>mMinEnergy)
  {
    G4PrimaryParticle* pParticle = new G4PrimaryParticle(pParticleDefinition);
  //    G4PrimaryParticle* pParticle = new G4PrimaryParticle(G4Gamma::Gamma());
    pParticle->SetTotalEnergy(energy);

    // look up range from histogram and add to the position
    position = m_posSPS->GenerateOne();
    direction = m_angSPS->GenerateOne();
    position += GetRange(energy) * direction;

    // look up angular offset (offset between direction and momentum) and adjust direction
    momentum_dir = PerturbVector(direction,GetAngle(energy));
    pParticle->SetMomentumDirection(momentum_dir);

    G4PrimaryVertex* pVertex = new G4PrimaryVertex(position, m_time);
    pVertex->SetPrimary(pParticle);

    event->AddPrimaryVertex(pVertex);

    if(nVerboseLevel>1)
    {
      G4cout << "GateSourceY90Brem::GeneratePrimaries()\n";
      G4cout << "Energy: " << mEnergy << " MeV\n";
      G4cout << "Direction: (" << direction.getX() << "," << direction.getY() << "," << direction.getZ() << ")\n";
      G4cout << "Position: (" << position.getX() << "," << position.getY() << "," << position.getZ() << ")\n";
      G4cout << "Momentum direction: (" << momentum_dir.getX() << "," << momentum_dir.getY() << "," << momentum_dir.getZ() << ")\n";
      G4cout << "Relative angle: " << direction.angle(momentum_dir) << G4endl;
    }
    numVertices++;
  }
  else
    if(nVerboseLevel>1)
      G4cout << "Energy below Emin. No particle generated." << G4endl;

  return numVertices;
}

void GateSourceY90Brem::GeneratePrimaryVertex(G4Event*)
{
  ;
}

G4double GateSourceY90Brem::GetEnergy()
{
  int i=0;

  G4double energy;

  G4double P = G4UniformRand();
  while(P > mCumulativeEnergyTable[i] && i<200)
    i++;
  energy = (i + G4UniformRand())*10*keV;

  return energy;
}

G4double GateSourceY90Brem::GetRange(G4double energy)
{
  int bin;
  int i=0;
  bin = int(energy/0.02); //TODO: un-hardcode the energy bin widths?
  bin = min(bin,99);

  G4double range;

  G4float P = G4UniformRand();
  while(P > mCumulativeRangeTable[bin][i])
    i++;

  range = (i+ G4UniformRand())*0.1*mm;

  return range;
}

G4double GateSourceY90Brem::GetAngle(G4double energy)
{
  int bin;
  int i=0;
  bin = int(energy/0.02); //TODO: un-hardcode the energy bin widths?
  bin = min(bin,99);

  G4double angle;

  G4float P = G4UniformRand();
  while(P > mAngleTable[bin][i])
    i++;

  angle = (i+ G4UniformRand()) * CLHEP::pi / 180.0;

  return angle;
}

G4ThreeVector GateSourceY90Brem::PerturbVector(G4ThreeVector original, G4double theta)
{
  // calculate the unit vectors
  G4ThreeVector r_hat = original;
  G4ThreeVector theta_hat;
  G4ThreeVector phi_hat;
  G4double sin_theta = sqrt(1-r_hat.getZ()*r_hat.getZ()); // not the same theta, this is the theta of the unit vectors
  if(sin_theta==0)
  {
    theta_hat.set(r_hat.getZ(), 0, 0);
    phi_hat.set(0,1,0);
  }
  else
  {
    theta_hat.set(r_hat.getZ()*r_hat.getX()/sin_theta,r_hat.getZ()*r_hat.getY()/sin_theta,-sin_theta);
    phi_hat.set(-r_hat.getY()/sin_theta,r_hat.getX()/sin_theta,0);
  }

  // tilt the original vector by creating a new vector composed of the original unit vectors
  G4double phi = G4RandFlat::shoot(0.0,CLHEP::twopi);
  G4ThreeVector theVector = cos(theta)*r_hat + sin(theta)*cos(phi)*theta_hat + sin(theta)*sin(phi)*phi_hat;

  return theVector;
}

G4double GateSourceY90Brem::GetNextTime( G4double timeStart )
{

  // returns the proposed time for the next event of this source, sampled from the
  // source time distribution
  G4double aTime = DBL_MAX;

  if( m_activity > 0. )
    {
      // compute the present activity, on the base of the starting activity and the lifetime (if any)
      G4double activityNow = 0;
      if( timeStart < m_startTime )
        activityNow = 0.;
      else
      {
        // Force life time to 0, time is managed by GATE not G4
        //GetParticleDefinition()->SetPDGLifeTime(0);
        if( m_forcedUnstableFlag )
        {
          if( m_forcedLifeTime > 0. )
          {
            activityNow = m_activity *
                exp( - ( timeStart - m_startTime ) / m_forcedLifeTime );
          }
          else
          {
            G4cout << "[GateSourceY90Brem::GetNextTime] ERROR: Forced decay with negative lifetime: (s) "
                << m_forcedLifeTime/s << Gateendl;
          }
        }
        else
          activityNow = m_activity;
      }
      if( nVerboseLevel > 0 )
        G4cout << "GateSourceY90Brem::GetNextTime : Initial activity (becq) : "
        << m_activity/becquerel << Gateendl
        << "                            At time (s) " << timeStart/s
        << " activity (becq) " << activityNow/becquerel << Gateendl;

      // sampling of the interval distribution
      if (!mEnableRegularActivity)
      {
        aTime = -log( G4UniformRand() ) * ( 1. / (mBremProb * activityNow )); // activity is reduced here
      }
      else {
        GateError("I should not be here. ");
        aTime = 1./activityNow;
      }
    }

  if( nVerboseLevel > 0 )
    G4cout << "GateSourceY90Source::GetNextTime : next time (s) " << aTime/s << Gateendl;

  return aTime;
}
