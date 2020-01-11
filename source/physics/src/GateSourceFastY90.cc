#include "GateSourceFastY90.hh"
#include "GateVoxelizedPosDistribution.hh"

GateSourceFastY90::GateSourceFastY90(G4String name) : GateVSource( name )
{
  GateMessage("Beam", 1, "GateSourceFastY90::GateSourceFastY90(G4String) called." << Gateendl);

  m_sourceMessenger = new GateSourceFastY90Messenger(this);

  int i, j;
  std::ifstream input_file;

  mPosProb = 3.186e-5;
  mGammaProb = 0.0;

  mCumulativeEnergyTable = new G4double[200];
  mMinEnergy = 0.0;
  CalculateEnergyTable();

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
  pGammaParticleDefinition = particleTable->FindParticle("gamma");
  pPositronParticleDefinition = particleTable->FindParticle("e+");

  m_angSPS->SetAngDistType("iso");

}

GateSourceFastY90::~GateSourceFastY90()
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

G4int GateSourceFastY90::GeneratePrimaries(G4Event *event)
{

  G4int numVertices = 0;
  G4PrimaryParticle* pParticle;
  G4double energy;
  G4ThreeVector position;
  G4ThreeVector direction;      // direction vector from beta particle source to point of brem
  G4ThreeVector momentum_dir;

  G4PrimaryVertex* pVertex;

  GateMessage("Beam", 1, "GateSourceFastY90::GeneratePrimaries(G4Event*) called at " << m_time << " s." << Gateendl);
  SetParticleTime(m_time);

  // get the particle position
  position = m_posSPS->GenerateOne();
  direction = m_angSPS->GenerateOne();

  // relative position if attached
  ChangeParticlePositionRelativeToAttachedVolume(position);

  G4double P = G4UniformRand();
  if (P < (mPosProb/(mPosProb+mBremProb)) ) // generate positron;
  {
    pParticle = new G4PrimaryParticle(pPositronParticleDefinition);
    energy = GetPositronEnergy();
    pParticle->SetKineticEnergy(energy);

    pParticle->SetMomentumDirection(direction);

    pVertex = new G4PrimaryVertex(position, m_time);
    pVertex->SetPrimary(pParticle);

    event->AddPrimaryVertex(pVertex);

    if(nVerboseLevel>1)
    {
      G4cout << "GateSourceFastY90::GeneratePrimaries()\n";
      G4cout << "++ positron ++\n";
      G4cout << "Energy: " << energy << " MeV\n";
      G4cout << "Direction: (" << direction.getX() << "," << direction.getY() << "," << direction.getZ() << ")\n";
      G4cout << "Position: (" << position.getX() << "," << position.getY() << "," << position.getZ() << ")\n";
    }
    numVertices++;

  } else {
    // generate brem
    energy = GetBremsstrahlungEnergy();
    pParticle = new G4PrimaryParticle(pGammaParticleDefinition);
    //    G4PrimaryParticle* pParticle = new G4PrimaryParticle(G4Gamma::Gamma());
    pParticle->SetTotalEnergy(energy);

    // look up range from histogram and add to the position
    position += GetRange(energy) * direction;

    // look up angular offset (offset between direction and momentum) and adjust direction
    G4double delta_angle = GetAngle(energy);
    momentum_dir = PerturbVector(direction,delta_angle);
    pParticle->SetMomentumDirection(momentum_dir);

    pVertex = new G4PrimaryVertex(position, m_time);
    pVertex->SetPrimary(pParticle);

    event->AddPrimaryVertex(pVertex);

    if(nVerboseLevel>1)
    {
      G4cout << "GateSourceFastY90::GeneratePrimaries()\n";
      G4cout << "++ brem photon ++\n";
      G4cout << "Energy: " << energy << " MeV\n";
      G4cout << "Direction: (" << direction.getX() << "," << direction.getY() << "," << direction.getZ() << ")\n";
      G4cout << "Position: (" << position.getX() << "," << position.getY() << "," << position.getZ() << ")\n";
      G4cout << "Momentum direction: (" << momentum_dir.getX() << "," << momentum_dir.getY() << "," << momentum_dir.getZ() << ")\n";
      G4cout << "Relative angle: " << delta_angle << G4endl;
    }
    numVertices++;
  }

  return numVertices;
}

void GateSourceFastY90::GeneratePrimaryVertex(G4Event*)
{
  ;
}

void GateSourceFastY90::CalculateEnergyTable()
{
  G4int i;
  G4int firstBin;

  // The actual minimum energy is going to fall on a 10 keV boundary. Handling this more exactly
  // isn't worth the effort, since this is a cutoff to get rid of photons that aren't going anywhere,
  // not a precisely calculated point
  firstBin = G4int( mMinEnergy / (10*keV) );

  for(i=0;i<firstBin;i++)
    mCumulativeEnergyTable[i]  = 0;
  mCumulativeEnergyTable[firstBin] = mEnergyTable[firstBin];
  for(i=firstBin+1;i<200;i++)
    mCumulativeEnergyTable[i]=mCumulativeEnergyTable[i-1] + mEnergyTable[i];

  for(i=0;i<200;i++)
    mCumulativeEnergyTable[i] /= mCumulativeEnergyTable[199];

  // calculate total probability of bremsstrahlung emission for a single beta
  mBremProb = mEnergyTable[firstBin];
  for(i=firstBin+1;i<200;i++)
    mBremProb += mEnergyTable[i];

  G4cout << "Total brem prob: " << mBremProb << G4endl;

}


G4double GateSourceFastY90::GetBremsstrahlungEnergy()
{
  int i=0;

  G4double energy;

  G4double P = G4UniformRand();
  while(P > mCumulativeEnergyTable[i] && i<200)
    i++;
  energy = (i + G4UniformRand())*(10*keV);

  return energy;
}

G4double GateSourceFastY90::GetPositronEnergy()
{
  int i=0;
  G4double energy;
  G4double P=G4UniformRand();
  while(P > mPositronEnergyTable[i])
    i++;
  energy = i * keV;
  return energy;
}

G4double GateSourceFastY90::GetRange(G4double energy)
{
  int bin;
  int i=0;
  bin = int(energy/(20.0*keV)); //TODO: un-hardcode the energy bin widths?
  bin = std::min(bin,99);

  G4double range;

  G4float P = G4UniformRand();
  while(P > mCumulativeRangeTable[bin][i])
    i++;

  range = (i+ G4UniformRand())*(0.1*mm);

  return range;
}

G4double GateSourceFastY90::GetAngle(G4double energy)
{
  int bin;
  int i=0;
  bin = int(energy/(20.0*keV)); //TODO: un-hardcode the energy bin widths?
  bin = std::min(bin,99);

  G4double angle;

  G4float P = G4UniformRand();
  while(P > mCumulativeAngleTable[bin][i])
    i++;

  angle = (i+ G4UniformRand()) * CLHEP::pi / 180.0;

  return angle;
}

G4ThreeVector GateSourceFastY90::PerturbVector(G4ThreeVector original, G4double alpha)
{
  // calculate the unit vectors
  G4ThreeVector r_hat = original;
  G4ThreeVector theta_hat;
  G4ThreeVector phi_hat;
  G4double sin_theta = sqrt(1-r_hat.getZ()*r_hat.getZ());
  if(sin_theta==0)
  {
    theta_hat.set(1, 0, 0);
    phi_hat.set(0,1,0);
  }
  else
  {
    theta_hat.set(r_hat.getZ()*r_hat.getX()/sin_theta,r_hat.getZ()*r_hat.getY()/sin_theta,-sin_theta);
    phi_hat.set(-r_hat.getY()/sin_theta,r_hat.getX()/sin_theta,0);
  }

  // tilt the original vector by creating a new vector composed of the original unit vectors
  G4double phi = G4RandFlat::shoot(0.0,CLHEP::twopi);
  G4ThreeVector theVector = cos(alpha)*r_hat + sin(alpha)*cos(phi)*theta_hat + sin(alpha)*sin(phi)*phi_hat;

  return theVector;
}

G4double GateSourceFastY90::GetNextTime( G4double timeStart )
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
        aTime = -log( G4UniformRand() ) * ( 1. / ((mPosProb + mBremProb) * activityNow )); // activity is reduced here
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

void GateSourceFastY90::LoadVoxelizedPhantom(G4String filename)
{
  if(m_posSPS)
    delete m_posSPS;
  m_posSPS = new GateVoxelizedPosDistribution(filename);
  m_angSPS->SetPosDistribution(m_posSPS);

}

void GateSourceFastY90::SetPhantomPosition(G4ThreeVector pos)
{
  GateVoxelizedPosDistribution* posDist = dynamic_cast<GateVoxelizedPosDistribution*>(m_posSPS);
  if(posDist)
    posDist->SetPosition(pos);
  else
    G4cout << "Can't use this command unless a voxelized phantom has already been loaded." << G4endl;

}
