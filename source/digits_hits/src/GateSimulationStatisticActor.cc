/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*
  \brief Class GateSimulationStatisticActor :
  \brief
*/

#ifndef GATESIMULATIONSTATISTICACTOR_CC
#define GATESIMULATIONSTATISTICACTOR_CC

#include "GateSimulationStatisticActor.hh"
#include "GateMiscFunctions.hh"
#include "GateApplicationMgr.hh"
#include "G4Event.hh"

double get_elapsed_time(const timeval &start, const timeval &end) {
    double elapsed = 0;
    elapsed += end.tv_sec + 1e-6 * end.tv_usec;
    elapsed -= start.tv_sec + 1e-6 * start.tv_usec;
    return elapsed;
}

std::string get_date_string() {
    time_t now = time(NULL);
    return std::string(ctime(&now));
}


//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateSimulationStatisticActor::GateSimulationStatisticActor(G4String name, G4int depth) :
    GateVActor(name, depth) {
    GateDebugMessageInc("Actor", 4, "GateSimulationStatisticActor() -- begin\n");
    //SetTypeName("SimulationStatisticActor");
    ResetData();
    GateDebugMessageDec("Actor", 4, "GateSimulationStatisticActor() -- end\n");
    gettimeofday(&start, NULL);
    startDateStr = get_date_string();
    mTrackTypesFlag = false;
    pMessenger = new GateSimulationStatisticActorMessenger(this);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateSimulationStatisticActor::~GateSimulationStatisticActor() {
    delete pMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateSimulationStatisticActor::Construct() {
    GateVActor::Construct();
    // Enable callbacks
    EnableBeginOfRunAction(true);
    EnableBeginOfEventAction(true);
    EnablePreUserTrackingAction(true);
    EnableUserSteppingAction(true);
    ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Callback Begin of Run
void GateSimulationStatisticActor::BeginOfRunAction(const G4Run *r) {
    if (mNumberOfRuns == 0) { gettimeofday(&start_afterinit, NULL); }
    GateVActor::BeginOfRunAction(r);
    mNumberOfRuns++;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Callback Begin Event
void GateSimulationStatisticActor::BeginOfEventAction(const G4Event *e) {
    if (e->GetNumberOfPrimaryVertex() > 0) {
        GateVActor::BeginOfEventAction(e);
        mNumberOfEvents++;
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback Begin Track
void GateSimulationStatisticActor::PreUserTrackingAction(const GateVVolume *v, const G4Track *t) {
    GateVActor::PreUserTrackingAction(v, t);
    mNumberOfTrack++;
    if (mTrackTypesFlag) {
        auto name = t->GetParticleDefinition()->GetParticleName();
        mTrackTypes[name]++;
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callbacks
void GateSimulationStatisticActor::UserSteppingAction(const GateVVolume *v, const G4Step *step) {
    GateVActor::UserSteppingAction(v, step);
    mNumberOfSteps++;

    // Get if boundary is reach
    G4StepPoint *pPostStepP = step->GetPostStepPoint();
    if (pPostStepP->GetStepStatus() == fGeomBoundary) {
        mNumberOfGeometricalSteps++;
    } else {
        mNumberOfPhysicalSteps++;
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GateSimulationStatisticActor::SaveData() {
    GateVActor::SaveData();
    timeval end;
    gettimeofday(&end, NULL);
    std::ofstream os;
    OpenFileOutput(mSaveFilename, os);

    double currentSimulationTime = GateApplicationMgr::GetInstance()->GetCurrentTime();

    double virtualStartTime = GateApplicationMgr::GetInstance()->GetVirtualTimeStart();
    double virtualStopTime = GateApplicationMgr::GetInstance()->GetVirtualTimeStop();
    double startTime = GateApplicationMgr::GetInstance()->GetTimeStart();
    double stopTime = GateApplicationMgr::GetInstance()->GetTimeStop();

    double elapsedSimulationTime = currentSimulationTime - startTime;
    if (virtualStartTime != -1) elapsedSimulationTime = currentSimulationTime - virtualStartTime;

    double t = get_elapsed_time(start, end);
    double twi = get_elapsed_time(start_afterinit, end);

    os << "# NumberOfRun    = " << mNumberOfRuns << Gateendl
       << "# NumberOfEvents = " << mNumberOfEvents << Gateendl
       << "# NumberOfTracks = " << mNumberOfTrack << Gateendl
       << "# NumberOfSteps  = " << mNumberOfSteps << Gateendl
       << "# NumberOfGeometricalSteps  = " << mNumberOfGeometricalSteps << Gateendl
       << "# NumberOfPhysicalSteps     = " << mNumberOfPhysicalSteps << Gateendl
       << "# ElapsedTime           = " << t << Gateendl
       << "# ElapsedTimeWoInit     = " << twi << Gateendl
       << "# StartDate             = " << startDateStr
       << "# EndDate               = " << get_date_string()
       << "# StartSimulationTime        = " << startTime / s << Gateendl
       << "# StopSimulationTime         = " << stopTime / s << Gateendl
       << "# CurrentSimulationTime      = " << currentSimulationTime / s << Gateendl
       << "# VirtualStartSimulationTime = " << virtualStartTime / s << Gateendl
       << "# VirtualStopSimulationTime  = " << virtualStopTime / s << Gateendl
       << "# ElapsedSimulationTime      = " << elapsedSimulationTime / s << Gateendl
       << "# PPS (Primary per sec)      = " << mNumberOfEvents / twi << Gateendl
       << "# TPS (Track per sec)        = " << mNumberOfTrack / twi << Gateendl
       << "# SPS (Step per sec)         = " << mNumberOfSteps / twi << Gateendl;

    if (mTrackTypesFlag) {
        os << "# Track types: " << Gateendl;
        for (auto item:mTrackTypes) {
            os << "# " << item.first << " = " << item.second << Gateendl;
        }
    }

    if (!os) {
        GateMessage("Output", 1, "Error Writing file: " << mSaveFilename << Gateendl);
    }
    os.flush();
    os.close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateSimulationStatisticActor::ResetData() {
    mNumberOfRuns = 0;
    mNumberOfEvents = 0;
    mNumberOfTrack = 0;
    mNumberOfGeometricalSteps = 0;
    mNumberOfPhysicalSteps = 0;
    mNumberOfSteps = 0;
    gettimeofday(&start, NULL);
    startDateStr = get_date_string();
}
//-----------------------------------------------------------------------------


#endif /* end #define GATESIMULATIONSTATISTICACTOR_CC */
