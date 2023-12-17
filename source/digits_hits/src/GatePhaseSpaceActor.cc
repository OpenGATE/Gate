/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

//
/*
  \brief Class GatePhaseSpaceActor
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr
  brent.huisman@insa-lyon.fr
*/

#include "GatePhaseSpaceActor.hh"
#include <G4EmCalculator.hh>
#include <G4VProcess.hh>
#include <G4Run.hh>
#include "GateMiscFunctions.hh"

#include "GateSourceMgr.hh"
#include "GateVImageActor.hh"
#include "GateProtonNuclearInformationActor.hh"

#include "GateSourceTPSPencilBeam.hh"
#include "GatePhaseSpaceActorMessenger.hh"
#include "GateIAEAHeader.h"

// --------------------------------------------------------------------
GatePhaseSpaceActor::GatePhaseSpaceActor(G4String name, G4int depth) : GateVActor(name, depth)
{
    GateDebugMessageInc("Actor", 4, "GatePhaseSpaceActor() -- begin\n");

    pMessenger = new GatePhaseSpaceActorMessenger(this);
    EnableCharge = true;
    EnableAtomicNumber = true;
    EnableElectronicDEDX = false;
    EnableTotalDEDX = false;
    EnableXPosition = true;
    EnableYPosition = true;
    EnableZPosition = true;
    EnableEkine = true;
    EnableXDirection = true;
    EnableYDirection = true;
    EnableZDirection = true;
    EnablePartName = true;
    EnableProdVol = true;
    EnableProdProcess = true;
    EnableWeight = true;
    EnableTime = false;
    EnableIonTime = false;
    EnableLocalTime = false;
    EnableTimeFromBeginOfEvent = false;
    EnableMass = true;
    EnableSec = false;
    EnableNuclearFlag = false;
    mIsFirstStep = true;
    mUseVolFrame = false;
    mStoreOutPart = false;
    SetIsAllStep(false);
    EnableTOut = false;
    EnableTProd = false;
    EnableTrackLengthFlag = false;

    mSphereProjectionFlag = false;
    mSphereProjectionCenter = G4ThreeVector(0);
    mSphereProjectionRadius = 0.0;

    mTranslateAlongDirectionFlag = false;
    mTranslationLength = 0.0;

    bEnableCoordFrame = false;
    bEnablePrimaryEnergy = false;
    bEnableSpotID = false;
    bEnableCompact = false;
    bEnableEmissionPoint = false;
    bEnablePDGCode = false;
    bEnableTOut = true;
    bEnableTProd = true;

    bSpotID = 0;
    bSpotIDFromSource = " ";
    bCoordFrame = " ";
    mMaskFilename = "";
    mMaskIsEnabled = false;
    mKillParticleFlag = false;

    mFileType = " ";
    mNevent = 0;
    pIAEARecordType = 0;
    pIAEAheader = 0;
    mFileSize = 0;

    GateDebugMessageDec("Actor", 4, "GatePhaseSpaceActor() -- end\n");

    emcalc = new G4EmCalculator;
}
// --------------------------------------------------------------------

// --------------------------------------------------------------------
/// Destructor
GatePhaseSpaceActor::~GatePhaseSpaceActor()
{
    GateDebugMessageInc("Actor", 4, "~GatePhaseSpaceActor() -- begin\n");

    free(pIAEAheader);
    free(pIAEARecordType);
    pIAEAheader = 0;
    pIAEARecordType = 0;

    delete pMessenger;
    GateDebugMessageDec("Actor", 4, "~GatePhaseSpaceActor() -- end\n");
}
// --------------------------------------------------------------------

// --------------------------------------------------------------------
/// Construct
void GatePhaseSpaceActor::Construct()
{
    GateVActor::Construct();
    // Enable callbacks
    EnableBeginOfRunAction(true);
    EnableBeginOfEventAction(true);
    EnableRecordEndOfAcquisition(true);
    EnablePreUserTrackingAction(true);
    EnablePostUserTrackingAction(true);
    EnableUserSteppingAction(true);

    G4String extension = getExtension(mSaveFilename);

    // If mask, load the image
    if (mMaskIsEnabled)
    {
        GateMessage("Actor", 1, "GatePhaseSpaceActor read mask file " << mMaskFilename);
        mMask.Read(mMaskFilename);
    }

    if (extension == "IAEAphsp" || extension == "IAEAheader")
    {
        mFileType = "iaeaFile";
        pIAEAheader = (iaea_header_type *)calloc(1, sizeof(iaea_header_type));
        pIAEAheader->initialize_counters();
        pIAEARecordType = (iaea_record_type *)calloc(1, sizeof(iaea_record_type));

        G4String IAEAFileExt = ".IAEAphsp";
        G4String IAEAFileName = " ";
        IAEAFileName = G4String(removeExtension(mSaveFilename));

        pIAEARecordType->p_file = open_file(const_cast<char *>(IAEAFileName.c_str()),
                                            const_cast<char *>(IAEAFileExt.c_str()), (char *)"wb");

        if (pIAEARecordType->p_file == NULL)
            GateError("File " << IAEAFileName << IAEAFileExt << " not opened.");
        if (pIAEARecordType->initialize() != OK)
            GateError("File " << IAEAFileName << IAEAFileExt << " not initialized.");

        if (EnableXPosition)
            pIAEARecordType->ix = 1;
        if (EnableYPosition)
            pIAEARecordType->iy = 1;
        if (EnableZPosition)
            pIAEARecordType->iz = 1;
        if (EnableXDirection)
            pIAEARecordType->iu = 1;
        if (EnableYDirection)
            pIAEARecordType->iv = 1;
        if (EnableZDirection)
            pIAEARecordType->iw = 1;
        if (EnableWeight)
            pIAEARecordType->iweight = 1;
        if (EnableTime || EnableLocalTime || EnableIonTime)
        {
            GateWarning("'Time' is not available in IAEA phase space.");
        }
        if (EnableMass)
        {
            GateWarning("'Mass' is not available in IAEA phase space.");
        }
        if (pIAEAheader->set_record_contents(pIAEARecordType) == FAIL)
            GateError("Record contents not setted.");
    }
    else
    {
        if (extension == "root")
        {
            mFileType = "rootFile";
        }
        else if (extension == "npy")
        {
            mFileType = "npyFile";
        }
        else if (extension == "txt")
        {
            mFileType = "txtFile";
        }
        else
            GateError("Unknown extension for phasespace");
    }
}
// --------------------------------------------------------------------

void GatePhaseSpaceActor::InitTree()
{
    mFile = new GateOutputTreeFileManager();

    if (mFileType == "npyFile")
        mFile->add_file(mSaveFilename, "npy");
    if (mFileType == "rootFile")
        mFile->add_file(mSaveFilename, "root");
    if (mFileType == "txtFile")
        mFile->add_file(mSaveFilename, "txt");

    mFile->set_tree_name("PhaseSpace");

    if (EnableAtomicNumber)
        mFile->write_variable("AtomicNumber", &Za);

    if (EnableCharge)
        mFile->write_variable("Charge", &charge);

    if (EnableElectronicDEDX)
        mFile->write_variable("ElectronicDEDX", &elecDEDX);
    if (EnableElectronicDEDX)
        mFile->write_variable("StepLength", &stepLength);
    if (EnableElectronicDEDX)
        mFile->write_variable("Edep", &edep);
    if (EnableTotalDEDX)
        mFile->write_variable("TotalDEDX", &totalDEDX);

    if (EnableEkine)
        mFile->write_variable("Ekine", &e);
    if (EnableElectronicDEDX)
        mFile->write_variable("Ekpost", &ekPost);
    if (EnableElectronicDEDX)
        mFile->write_variable("Ekpre", &ekPre);
    if (EnableWeight)
        mFile->write_variable("Weight", &w);
    if (EnableTime || EnableLocalTime)
        mFile->write_variable("Time", &t);
    if (EnableIonTime)
        mFile->write_variable("IonTime", &pt);
    if (EnableMass)
        mFile->write_variable("Mass", &m); // in MeV/c2
    if (EnableXPosition)
        mFile->write_variable("X", &x);
    if (EnableYPosition)
        mFile->write_variable("Y", &y);
    if (EnableZPosition)
        mFile->write_variable("Z", &z);
    if (EnableXDirection)
        mFile->write_variable("dX", &dx);
    if (EnableYDirection)
        mFile->write_variable("dY", &dy);
    if (EnableZDirection)
        mFile->write_variable("dZ", &dz);

    if (EnableTrackLengthFlag)
        mFile->write_variable("trackLength", &trackLength);

    if (EnablePartName /*&& bEnableCompact==false*/)
        mFile->write_variable("ParticleName", pname, sizeof(pname));
    if (EnableProdVol && bEnableCompact == false)
        mFile->write_variable("ProductionVolume", vol, sizeof(vol));
    if (EnableProdProcess && bEnableCompact == false)
        mFile->write_variable("CreatorProcess", creator_process, sizeof(creator_process));
    if (EnableProdProcess && bEnableCompact == false)
        mFile->write_variable("ProcessDefinedStep", pro_step, sizeof(pro_step));
    if (bEnableCompact == false)
        mFile->write_variable("TrackID", &trackid);
    if (bEnableCompact == false)
        mFile->write_variable("ParentID", &parentid);
    if (bEnableCompact == false)
        mFile->write_variable("EventID", &eventid);
    if (bEnableCompact == false)
        mFile->write_variable("RunID", &runid);
    if (bEnablePrimaryEnergy)
        mFile->write_variable("PrimaryEnergy", &bPrimaryEnergy);
    if (bEnablePDGCode)
        mFile->write_variable("PDGCode", &bPDGCode);
    if (bEnableEmissionPoint)
    {
        mFile->write_variable("EmissionPointX", &bEmissionPointX);
        mFile->write_variable("EmissionPointY", &bEmissionPointY);
        mFile->write_variable("EmissionPointZ", &bEmissionPointZ);
    }
    if (bEnableSpotID)
        mFile->write_variable("SpotID", &bSpotID);

    if (EnableTOut)
        mFile->write_variable("TOut", &tOut);
    if (EnableTProd)
        mFile->write_variable("TProd", &tProd);

    if (EnableTimeFromBeginOfEvent)
        mFile->write_variable("TimeFromBeginOfEvent", &fTimeFromBeginOfEvent);

    if (EnableNuclearFlag)
    {
        mFile->write_variable("CreatorProcess", &creator);
        mFile->write_variable("NuclearProcess", &nucprocess);
        mFile->write_variable("Order", &order);
    }
    mFile->write_header();
}

// --------------------------------------------------------------------
void GatePhaseSpaceActor::BeginOfRunAction(const G4Run *r)
{
    if (!this->mOverWriteFilesFlag)
    {
        mSaveFilename = GetSaveCurrentFilename(mSaveInitialFilename);
        InitTree();
    }
    else
    {
        // Only init the tree at the first run
        if (r->GetRunID() == 0)
        {
            InitTree();
        }
    }
}
// --------------------------------------------------------------------

// --------------------------------------------------------------------
void GatePhaseSpaceActor::RecordEndOfAcquisition()
{
    // For npy output, write and close must be done at the end.
    if (this->mOverWriteFilesFlag)
    {
        mFile->write();
        mFile->close();
    }
}
// --------------------------------------------------------------------

// --------------------------------------------------------------------
void GatePhaseSpaceActor::SetMaskFilename(G4String filename)
{
    mMaskFilename = filename;
    mMaskIsEnabled = true;
}
// --------------------------------------------------------------------

// --------------------------------------------------------------------
void GatePhaseSpaceActor::SetKillParticleFlag(bool b)
{
    mKillParticleFlag = b;
}
// --------------------------------------------------------------------

// --------------------------------------------------------------------
void GatePhaseSpaceActor::PreUserTrackingAction(const GateVVolume * /*v*/, const G4Track *t)
{
    mIsFirstStep = true;
    if (bEnableEmissionPoint)
    {
        bEmissionPointX = t->GetVertexPosition().x();
        bEmissionPointY = t->GetVertexPosition().y();
        bEmissionPointZ = t->GetVertexPosition().z();
    }
}
// --------------------------------------------------------------------

// --------------------------------------------------------------------
void GatePhaseSpaceActor::PostUserTrackingAction(const GateVVolume * /*v*/, const G4Track *t)
{
    if (EnableTrackLengthFlag)
    {
        trackLength = t->GetTrackLength();
    }
}
// --------------------------------------------------------------------

// --------------------------------------------------------------------
void GatePhaseSpaceActor::BeginOfEventAction(const G4Event *e)
{
    // Set Primary Energy
    bPrimaryEnergy = e->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy(); // GetInitialEnergy oid.
    if (EnableIonTime) pt = e->GetPrimaryVertex()->GetT0();

    // Store the application time of the event
    auto app = GateApplicationMgr::GetInstance();
    fBeginOfEventTime = app->GetCurrentTime();
    // std::cout << "GatePhaseSpaceActor::BeginOfEventAction fBeginOfEventTime = "
    //           << G4BestUnit(fBeginOfEventTime, "Time") << std::endl;

    // Set SourceID
    if (GetIsSpotIDEnabled())
    {
        GateSourceTPSPencilBeam *tpspencilsource =
            dynamic_cast<GateSourceTPSPencilBeam *>(GateSourceMgr::GetInstance()->GetSourceByName(
                bSpotIDFromSource));
        bSpotID = tpspencilsource->GetCurrentSpotID();
    }
}
// --------------------------------------------------------------------

// --------------------------------------------------------------------
void GatePhaseSpaceActor::UserSteppingAction(const GateVVolume *, const G4Step *step)
{

    /*
      Options:
      - storeOutgoingParticles --> consider PostPoint instead of PrePoint
      - storeAllStep           --> store every step, not only the first time it enters the volume
      - storeSecondaries       --> also store particle created inside the volume
      - attachMask             --> store particle only when it is in the mask
      - killParticle           --> kill the particle once stored
    */

    // IsFirstStep is true when this is the first time the track is in this function
    // It is initialized to true in PreUserTrackingAction and to false here.

    // If track already stored and the option EnableAllStep is not true, do nothing.
    if (!mIsFirstStep && !EnableAllStep)
        return;

    // If this is the first time we see the track and this is an event,
    // increment mNevent (used by iaea output)
    if (mIsFirstStep && step->GetTrack()->GetTrackID() == 1)
        mNevent++;

    // By default, we consider the preStep (starting point), except if
    // user set mStoreOutPart or if the option EnableAllStep is enabled
    G4StepPoint *stepPoint;
    if (mStoreOutPart || EnableAllStep)
        stepPoint = step->GetPostStepPoint();
    else
        stepPoint = step->GetPreStepPoint();

    // If needed, check if in the mask, if not, do nothing (do not store, do not mark particle)
    if (mMaskIsEnabled)
    {
        // auto vox = dynamic_cast<const GateVImageVolume*>(mVolume);
        if (mVolume != NULL)
        {
            const bool mPositionIsSet = false;
            const G4ThreeVector mPosition;
            const GateVImageActor::StepHitType mStepHitType = GateVImageActor::PreStepHitType;
            auto index = GateVImageActor::GetIndexFromStepPosition2(mVolume, step, mMask, mPositionIsSet, mPosition,
                                                                    mStepHitType);
            auto value = mMask.GetValue(index);
            if (value == 1)
                return; // do nothing
        }
    }

    //----------- Write volume name -------------
    G4String st = "";
    if (step->GetTrack()->GetLogicalVolumeAtVertex())
        st = step->GetTrack()->GetLogicalVolumeAtVertex()->GetName();
    strcpy(vol, st.c_str());

    //----------- ??? -------------
    // FIXME: Document what this is/does.
    // if(vol!=mVolume->GetLogicalVolumeName() && mStoreOutPart) return;
    if (vol == mVolume->GetLogicalVolumeName() && !EnableSec && !mStoreOutPart)
        return;
    // if(!( mStoreOutPart && step->IsLastStepInVolume())) return;

    //----------- ??? -------------
    // FIXME: Document what this is/does.
    // something wrong here:
    if (mStoreOutPart && step->GetTrack()->GetVolume() == step->GetTrack()->GetNextVolume())
        return;

    //----------- Workaround for outgoing particles flag -------------
    // FIXME: Document why necesary?
    if (mStoreOutPart)
    {
        /* 2014-06-11: Brent & David
         * There is a rare bug when using the PhaseSpaceActor to store outgoing particles and very long cuts on particles (nongammas).
         * When a particle crosses from a segmented_log_X volume to a segmented_log_X, Gate segfaults.
         * Seems that checking for null on pv and nextvol allows to program to complete.
         * Unsure if this hack is dirty and needs to be checked.
         */
        G4VPhysicalVolume *pv = step->GetTrack()->GetNextVolume();
        if (pv == 0)
            return;
        GateVVolume *nextVol = GateObjectStore::GetInstance()->FindVolumeCreator(pv);
        if (nextVol == 0)
            return;
        if (nextVol == mVolume)
            return;
        GateVVolume *parent = nextVol->GetParentVolume();
        while (parent)
        {
            if (parent == mVolume)
                return;
            parent = parent->GetParentVolume();
        }
    }

    //----------- ??? -------------
    // FIXME: remove?
    /*if(mStoreOutPart && step->GetTrack()->GetVolume()!=mVolume->GetPhysicalVolume() ){
      GateVVolume *parent = mVolume->GetParentVolume();
      while(parent){
      if(parent==mVolume) return;
      parent = parent->GetParentVolume();
      }
      }
    */

    //-----------Write name of the particles presents at the simulation-------------
    st = step->GetTrack()->GetDefinition()->GetParticleName();

    //'st' contains some nonprinteble caracters, which are not always the same. e.g. there exist multiple kinds of gammas, oxygens, etc.
    strcpy(pname, st.c_str());
    bPDGCode = step->GetTrack()->GetDefinition()->GetPDGEncoding();

    // cout << step->GetTrack()->GetDefinition()->GetPDGEncoding() << endl;
    //  TODO doesnt work, undefined reference. Problem with makefile?
    // Solution, use PDGcode instead of ParticleName. However, GatePhaseSpaceSource uses Particlename char[64] while GatePhaseSpaceActor stores Char_t[256].

    //------------Write position of the steps presents at the simulation-------------
    G4ThreeVector localPosition = stepPoint->GetPosition();

    if (GetUseVolumeFrame())
    {
        const G4AffineTransform transformation = step->GetPreStepPoint()->GetTouchable()->GetHistory()->GetTopTransform();
        localPosition = transformation.TransformPoint(localPosition);
    }
    else if (GetEnableCoordFrame())
    {
        // Give GetUseVolumeFrame preference

        // Find the transform from GetCoordFrame volume to the world.
        GateVVolume *v = GateObjectStore::GetInstance()->FindCreator(GetCoordFrame());
        if (v == NULL)
        {
            if (mFileType == "rootFile")
            {
                mFile->close();
            }
            GateError("Error, cannot find the volume '" << GetCoordFrame() << "' -> (see the setCoordinateFrame)");
        }

        G4VPhysicalVolume *phys = v->GetPhysicalVolume();
        G4AffineTransform volumeToWorld = G4AffineTransform(phys->GetRotation(), phys->GetTranslation());
        while (v->GetLogicalVolumeName() != "world_log")
        {
            v = v->GetParentVolume();
            phys = v->GetPhysicalVolume();
            G4AffineTransform x(phys->GetRotation(), phys->GetTranslation());
            volumeToWorld = volumeToWorld * x;
        }

        volumeToWorld = volumeToWorld.NetRotation();
        G4AffineTransform worldToVolume = volumeToWorld.Inverse();

        // old crap:stepLength
        // const G4AffineTransform transformation = GateObjectStore::GetInstance()->FindCreator(GetCoordFrame())->GetPhysicalVolume()->GetTouchable()->GetHistory()->GetTopTransform();
        localPosition = worldToVolume.TransformPoint(localPosition);
    }

    trackid = step->GetTrack()->GetTrackID();
    parentid = step->GetTrack()->GetParentID();
    eventid = GateRunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
    runid = GateRunManager::GetRunManager()->GetCurrentRun()->GetRunID();

    x = localPosition.x();
    y = localPosition.y();
    z = localPosition.z();

    //===============================================================================================================
    // ctq: Flags for proton nuclear processes
    // no process = 0
    // hadElastic = 1
    // protonInelastic = 2
    //===============================================================================================================

    if (EnableNuclearFlag)
    {
        GateProtonNuclearInformation *info = dynamic_cast<GateProtonNuclearInformation *>(step->GetTrack()->GetUserInformation());
        if (info == NULL)
            GateWarning("Could not retrieve GateProtonNuclearInformation, EnableNuclearFlag needs it");
        else
        {
            creator = 0;
            nucprocess = 0;
            order = info->GetScatterOrder();

            if (!step->GetTrack()->GetCreatorProcess())
                creator = 0;

            if (step->GetTrack()->GetCreatorProcess() &&
                step->GetTrack()->GetCreatorProcess()->GetProcessName() == "hadElastic")
                creator = 1;

            if (step->GetTrack()->GetCreatorProcess() &&
                step->GetTrack()->GetCreatorProcess()->GetProcessName() == "protonInelastic")
                creator = 2;

            if (!info->GetScatterProcess())
                nucprocess = 0;

            if (info->GetScatterProcess() == "hadElastic")
                nucprocess = 1;

            if (info->GetScatterProcess() == "protonInelastic")
                nucprocess = 2;
        }
    }

    // particle momentum
    // pc = sqrt(Ek^2 + 2*Ek*m_0*c^2)
    // sqrt( p*cos(Ax)^2 + p*cos(Ay)^2 + p*cos(Az)^2 ) = p

    //--------------Write momentum of the steps presents at the simulation----------
    G4ThreeVector localMomentum = stepPoint->GetMomentumDirection();

    if (GetUseVolumeFrame())
    {
        const G4AffineTransform transformation = step->GetPreStepPoint()->GetTouchable()->GetHistory()->GetTopTransform();
        localMomentum = transformation.TransformAxis(localMomentum);
    }
    else if (GetEnableCoordFrame())
    {
        // Give GetUseVolumeFrame preference

        // Find the transform from GetCoordFrame volume to the world.
        GateVVolume *v = GateObjectStore::GetInstance()->FindCreator(GetCoordFrame());
        G4VPhysicalVolume *phys = v->GetPhysicalVolume();
        G4AffineTransform volumeToWorld = G4AffineTransform(phys->GetRotation(), phys->GetTranslation());
        while (v->GetLogicalVolumeName() != "world_log")
        {
            v = v->GetParentVolume();
            phys = v->GetPhysicalVolume();
            G4AffineTransform x(phys->GetRotation(), phys->GetTranslation());
            volumeToWorld = volumeToWorld * x;
        }

        volumeToWorld = volumeToWorld.NetRotation();
        G4AffineTransform worldToVolume = volumeToWorld.Inverse();

        localMomentum = worldToVolume.TransformAxis(localMomentum);
    }

    dx = localMomentum.x();
    dy = localMomentum.y();
    dz = localMomentum.z();

    // time from the production to the leaving of the volume. Useful only for the outgoing particles
    tOut = step->GetTrack()->GetLocalTime();

    // production time of the particle wrt to the primary production (defined as a GlobalTime - LocalTime)"
    tProd = step->GetTrack()->GetGlobalTime() - (step->GetTrack()->GetLocalTime());

    // time from the time at event creation (could be different from the globaltime-localtime as globaltime is the
    fTimeFromBeginOfEvent = step->GetTrack()->GetGlobalTime() - fBeginOfEventTime;
    /*
    std::cout << "Step " << GetName() << " "
              << GateRunManager::GetRunManager()->GetCurrentEvent()->GetEventID() << " "
              << "global = " << G4BestUnit(step->GetTrack()->GetGlobalTime(), "Time")
              << " fbeg = " << G4BestUnit(fBeginOfEventTime, "Time")
              << " timefrom = " << G4BestUnit(fTimeFromBeginOfEvent, "Time")
              << std::endl;
              */

    //------------- Option to project position on a sphere
    /* Sometimes it is useful to store the particle position on a different
       position than the one where it has been detected. The option
       ''SphereProjection' change the particle position: it compute the projection
       on a sphere. */
    if (mSphereProjectionFlag)
    {
        // Project point position on the use sphere
        // https://en.wikipedia.org/wiki/Line–sphere_intersection

        // Use the notation of wikipedia (wikipedia rocks!)
        G4ThreeVector c = mSphereProjectionCenter;
        G4ThreeVector o(x, y, z);
        G4ThreeVector l(dx, dy, dz);
        double r = mSphereProjectionRadius;

        // Split equation in three parts A,B,C
        G4ThreeVector diff = o - c;
        double r2 = r * r;
        double A = -2 * (l.dot(o - c));
        double B = pow(2 * (l.dot(o - c)), 2) - 4 * l.dot(l) * (diff.dot(diff) - r2);
        double C = 2 * l.dot(l);

        // How many intersection ? 0,1 or 2 ?
        // If no intersection, we ignore this hit
        if (B < 0)
            return;

        // else we consider the closest one
        double d1 = (A + sqrt(B)) / C;
        double d2 = (A - sqrt(B)) / C;
        double d = d1;
        if (fabs(d2) < fabs(d1))
            d = d2;
        x = x + dx * d;
        y = y + dy * d;
        z = z + dz * d;
    }

    /*
      Translate the particle point along the direction by a given value
    */
    if (mTranslateAlongDirectionFlag)
    {

        // move point along its direction by a length d (may be negative)
        G4ThreeVector o(x, y, z);
        G4ThreeVector l(dx, dy, dz);

        x = x + dx * mTranslationLength;
        y = y + dy * mTranslationLength;
        z = z + dz * mTranslationLength;
    }

    //-------------Write weight of the steps presents at the simulation-------------
    w = stepPoint->GetWeight();

    if (EnableLocalTime)
    {
        t = stepPoint->GetLocalTime();
    }
    else
        t = stepPoint->GetGlobalTime();

    // t = step->GetTrack()->GetProperTime() ; //tibo : which time?????
    GateDebugMessage("Actor", 4, st << " stepPoint time proper=" << G4BestUnit(stepPoint->GetProperTime(), "Time") << " global=" << G4BestUnit(stepPoint->GetGlobalTime(), "Time") << " local=" << G4BestUnit(stepPoint->GetLocalTime(), "Time") << Gateendl);
    GateDebugMessage("Actor", 4, "trackid=" << step->GetTrack()->GetParentID() << " event=" << GateRunManager::GetRunManager()->GetCurrentEvent()->GetEventID() << " run=" << GateRunManager::GetRunManager()->GetCurrentRun()->GetRunID() << Gateendl);
    GateDebugMessage("Actor", 4, "pos = " << x << " " << y << " " << z << Gateendl);
    GateDebugMessage("Actor", 4, "E = " << G4BestUnit(stepPoint->GetKineticEnergy(), "Energy") << Gateendl);

    //---------Write energy of step present at the simulation--------------------------
    e = stepPoint->GetKineticEnergy();
    ekPost = step->GetPostStepPoint()->GetKineticEnergy();
    ekPre = step->GetPreStepPoint()->GetKineticEnergy();

    Za = step->GetTrack()->GetDefinition()->GetAtomicNumber(); // std::floor(stepPoint->GetCharge()+0.1);  //floor & +0.1 to avoid round off error
    charge = step->GetTrack()->GetDefinition()->GetPDGCharge();

    m = step->GetTrack()->GetDefinition()->GetAtomicMass();

    if (EnableElectronicDEDX || EnableTotalDEDX)
    {
        G4Material *material = step->GetPreStepPoint()->GetMaterial(); //->GetName();
        G4double energy1 = step->GetPreStepPoint()->GetKineticEnergy();
        G4double energy2 = step->GetPostStepPoint()->GetKineticEnergy();
        G4double energy = (energy1 + energy2) / 2;
        G4ParticleDefinition *partname = step->GetTrack()->GetDefinition(); //->GetParticleName();

        elecDEDX = emcalc->ComputeElectronicDEDX(energy, partname, material);
        stepLength = step->GetStepLength();

        edep = step->GetTotalEnergyDeposit() * w;
        totalDEDX = emcalc->ComputeTotalDEDX(energy, partname, material);
    }

    // elecDEDX= 1.;
    // totalDEDX=2.;

    // G4cout << st << " " << step->GetTrack()->GetDefinition()->GetAtomicMass() << " " << step->GetTrack()->GetDefinition()->GetPDGMass() << Gateendl;

    //----------Process name at origin Track--------------------
    st = "";
    if (step->GetTrack()->GetCreatorProcess())
        st = step->GetTrack()->GetCreatorProcess()->GetProcessName();
    strcpy(creator_process, st.c_str());

    //----------
    st = "";
    if (stepPoint->GetProcessDefinedStep())
        st = stepPoint->GetProcessDefinedStep()->GetProcessName();
    strcpy(pro_step, st.c_str());

    if (mFileType == "iaeaFile")
    {

        const G4Track *aTrack = step->GetTrack();
        int pdg = aTrack->GetDefinition()->GetPDGEncoding();

        if (pdg == 22)
            pIAEARecordType->particle = 1; // gamma
        else if (pdg == 11)
            pIAEARecordType->particle = 2; // electron
        else if (pdg == -11)
            pIAEARecordType->particle = 3; // positron
        else if (pdg == 2112)
            pIAEARecordType->particle = 4; // neutron
        else if (pdg == 2212)
            pIAEARecordType->particle = 5; // proton
        else
            GateError("Actor phase space: particle not available in IAEA format.");

        pIAEARecordType->energy = e;

        if (pIAEARecordType->ix > 0)
            pIAEARecordType->x = localPosition.x() / cm;
        if (pIAEARecordType->iy > 0)
            pIAEARecordType->y = localPosition.y() / cm;
        if (pIAEARecordType->iz > 0)
            pIAEARecordType->z = localPosition.z() / cm;

        if (pIAEARecordType->iu > 0)
            pIAEARecordType->u = localMomentum.x();
        if (pIAEARecordType->iv > 0)
            pIAEARecordType->v = localMomentum.y();
        if (pIAEARecordType->iw > 0)
            pIAEARecordType->w = fabs(localMomentum.z()) / localMomentum.z();

        // G4double charge = aTrack->GetDefinition()->GetPDGCharge();

        if (pIAEARecordType->iweight > 0)
            pIAEARecordType->weight = w;

        // pIAEARecordType->IsNewHistory = 0;  // not yet used

        pIAEARecordType->write_particle();

        pIAEAheader->update_counters(pIAEARecordType);
    } 
    else
    {
        mFile->fill();
//#define CORRELATION_TIME_ENERGY_SPACE
#ifdef CORRELATION_TIME_ENERGY_SPACE
	// std::cout << e << " " << tProd << std::endl;
	std::cout << e << " " << tProd << " " << x << " " << y << " " << z << std::endl;
#endif
    }
    mIsFirstStep = false;

    // Info will be stored so we marked the particle as killed
    if (mKillParticleFlag)
        step->GetTrack()->SetTrackStatus(fStopAndKill);
}
// --------------------------------------------------------------------

// --------------------------------------------------------------------
void GatePhaseSpaceActor::SaveData()
{
    GateVActor::SaveData();

    if (mFileType == "iaeaFile")
    {
        pIAEAheader->orig_histories = mNevent;
        G4String IAEAHeaderExt = ".IAEAheader";

        strcpy(pIAEAheader->title, "Phase space generated by GATE software (Geant4)");

        pIAEAheader->iaea_index = 0;

        G4String IAEAFileName = " ";
        IAEAFileName = G4String(removeExtension(mSaveFilename));
        pIAEAheader->fheader = open_file(const_cast<char *>(IAEAFileName.c_str()),
                                         const_cast<char *>(IAEAHeaderExt.c_str()), (char *)"wb");

        if (pIAEAheader->write_header() != OK)
            GateError("Phase space header not written.");

        fclose(pIAEAheader->fheader);
        fclose(pIAEARecordType->p_file);
    }
    else 
    {
        if (!this->mOverWriteFilesFlag)
        {
            // Write and close only whe we know that mFile will be recreated next run
            mFile->write();
            mFile->close();
        }
    }
}
// --------------------------------------------------------------------

// --------------------------------------------------------------------
void GatePhaseSpaceActor::ResetData()
{
    GateError("Can't reset phase space");
}
// --------------------------------------------------------------------
