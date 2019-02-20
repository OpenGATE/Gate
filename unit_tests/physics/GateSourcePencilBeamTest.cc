#include <catch.hpp>
#include <vector>
#include <cmath>
#include <string>
#include <numeric>
#include <GateSourcePencilBeam.hh>
#include <G4SystemOfUnits.hh>
#include <G4Event.hh>
#include <GateRunManager.hh>
#include <GatePhysicsList.hh>

double add_squares(double x,double y){return x+y*y;}
class varstat1D {
  private:
    std::string name_;
    std::vector<double> values_;
  public:
    varstat1D(std::string name="notset"):name_(name){ }
    void add_value(double val){values_.push_back(val);}
    double average(){ return std::accumulate(values_.begin(),values_.end(),0.)/values_.size(); }
    double rms(){
      double average = this->average();
      double average2 = std::accumulate(values_.begin(),values_.end(),0.,add_squares)/values_.size();
      double rms_ = std::sqrt(std::abs(average2-average*average));
      INFO( name_ << ": average=" << average << ", average of squares=" << average2 << " rms=" << rms_ );
      return rms_;
    }
    std::string name() const {return name_;}
    const std::vector<double>& get_data(){ return values_;};
};
class varstat2D {
  private:
    std::string name_;
    varstat1D xvar_;
    varstat1D yvar_;
    double xave_;
    double yave_;
    double xrms_;
    double yrms_;
    double rho_;
    bool up2date_;
    void cache(){
      if (up2date_) return;
      xave_ = xvar_.average();
      yave_ = yvar_.average();
      xrms_ = xvar_.rms();
      yrms_ = yvar_.rms();
      auto ix = xvar_.get_data().begin();
      auto iy = yvar_.get_data().begin();
      int n = 0;
      double xy = 0.;
      while (ix != xvar_.get_data().end()){
        xy *= n;
        xy += (*(ix++)-xave_)*(*(iy++)-yave_);
        xy /= (++n);
      }
      rho_ = xy / (xrms_ * yrms_);
      INFO( name_ << ": n=" << n << ", <xy>=" << xy << ", rho=" << rho_ );
      up2date_=true;
    }
  public:
    varstat2D(std::string xname,std::string yname):
        name_(xname+yname),xvar_(xname),yvar_(yname),
        xave_(NAN),yave_(NAN),xrms_(NAN),yrms_(NAN),rho_(NAN),up2date_(true){}
    void add_value(double xval, double yval){
      xvar_.add_value(xval);
      yvar_.add_value(yval);
      up2date_=false;
    }
    double averageX(){ cache(); return xave_; }
    double averageY(){ cache(); return yave_; }
    double rmsX(){ cache(); return xrms_; }
    double rmsY(){ cache(); return yrms_; }
    double rho(){ cache(); return rho_; }
};

TEST_CASE("pencil beam test","[example][physics]"){

  INFO("create geant4 environment for particle sources");
  auto runManager = new GateRunManager;
  runManager->SetUserInitialization( GatePhysicsList::GetInstance() );

  INFO("create pencil beam source");
  auto gpb = new GateSourcePencilBeam("test",false);

  INFO("set kinematics");
  gpb->SetIonParameter("proton");
  gpb->SetPosition(G4ThreeVector(10.*mm,20.*mm,5.*mm));
  gpb->SetSigmaX(1. * mm);
  gpb->SetSigmaY(1. * mm);
  gpb->SetSigmaTheta(1. * mrad);
  gpb->SetSigmaPhi(1. * mrad);
  gpb->SetEllipseXThetaArea(1.*mm*mrad);
  gpb->SetEllipseYPhiArea(1.*mm*mrad);
  gpb->SetEnergy(100 * MeV);
  gpb->SetSigmaEnergy(0.1 * MeV);

  varstat1D Estat("E");
  varstat2D XYstat("X","Y");
  varstat2D XDXstat("X","DX");
  varstat2D YDYstat("Y","DY");
  for (int i=0; i<1000; ++i){
    INFO("create event");
    auto event = new G4Event;

    INFO("generate primaries");
    G4int nret = gpb->GeneratePrimaries(event);

    INFO("number of generated primaries should be 1");
    REQUIRE(nret == 1);

    INFO("number of vertices in event should also be 1");
    G4int nvertex = event->GetNumberOfPrimaryVertex();
    REQUIRE(nvertex == 1);
    const G4PrimaryVertex *v = event->GetPrimaryVertex();
    double x = v->GetX0();
    double y = v->GetY0();
    double z = v->GetZ0();

    INFO("number of particles in vertex should also be 1");
    G4int nprimaries = v->GetNumberOfParticle();
    REQUIRE(nprimaries == 1);
    CHECK(z/mm == Approx(5.).margin(1e-10));

    const G4PrimaryParticle *p = v->GetPrimary();
    double dx = p->GetMomentumDirection()[0];
    double dy = p->GetMomentumDirection()[1];
    double dz = p->GetMomentumDirection()[2];
    CHECK(dx*dx+dy*dy+dz*dz == Approx(1.).margin(1.e-6));
    Estat.add_value( p->GetKineticEnergy() );
    XYstat.add_value(x,y);
    XDXstat.add_value(x,atan(dx));
    YDYstat.add_value(y,atan(dy));
    delete event;
  }
  INFO("energy spread should be 0.1 MeV");
  CHECK(Estat.rms()/MeV == Approx(0.1).margin(0.01));
  INFO("average energy should be 100 MeV within 0.1 sigma");
  CHECK(Estat.average()/MeV == Approx(100.).margin(0.01));
  INFO("rms X and Y should be 1 mm");
  CHECK(XYstat.rmsX()/mm == Approx(1.).margin(0.1));
  CHECK(XYstat.rmsY()/mm == Approx(1.).margin(0.1));
  INFO("average X and should be 10 mm and 20 mm within 0.1 sigma");
  CHECK(XYstat.averageX()/mm == Approx(10.).margin(0.1));
  CHECK(XYstat.averageY()/mm == Approx(20.).margin(0.1));
  INFO("rms theta and phi should be 1 mrad");
  CHECK(XDXstat.rmsY()/mrad == Approx(1.).margin(0.1));
  CHECK(YDYstat.rmsY()/mrad == Approx(1.).margin(0.1));
  INFO("average theta and phi should be 0 mrad");
  CHECK(XDXstat.averageY()/mrad == Approx(0.).margin(0.1));
  CHECK(YDYstat.averageY()/mrad == Approx(0.).margin(0.1));
  INFO("emittance in X (rho=" << XDXstat.rho() <<  ") and Y (rho="<< YDYstat.rho() << ")" );
  REQUIRE(pow(XDXstat.rho(),2.)<1);
  REQUIRE(pow(YDYstat.rho(),2.)<1);
  CHECK(pi*sqrt(1.-pow(XDXstat.rho(),2.)) == Approx(1.).margin(0.1));
  CHECK(pi*sqrt(1.-pow(YDYstat.rho(),2.)) == Approx(1.).margin(0.1));
}
