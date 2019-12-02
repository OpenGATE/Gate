#ifndef GATESTAT_H
#define GATESTAT_H

#include <catch.hpp>
#include <vector>
#include <cmath>
#include <string>
#include <numeric>

namespace unit_tests {

double add_squares(double x, double y);

class varstat1D {
private:
    std::string name_;
    std::vector<double> values_;
public:
    varstat1D(std::string name = "notset") : name_(name) {}

    void add_value(double val) { values_.push_back(val); }

    double average() { return std::accumulate(values_.begin(), values_.end(), 0.) / values_.size(); }

    double rms() {
        double average = this->average();
        double average2 = std::accumulate(values_.begin(), values_.end(), 0., add_squares) / values_.size();
        double rms_ = std::sqrt(std::abs(average2 - average * average));
        INFO(name_ << ": average=" << average << ", average of squares=" << average2 << " rms=" << rms_);
        return rms_;
    }

    std::string name() const { return name_; }

    const std::vector<double> &get_data() { return values_; };
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

    void cache() {
        if (up2date_) return;
        xave_ = xvar_.average();
        yave_ = yvar_.average();
        xrms_ = xvar_.rms();
        yrms_ = yvar_.rms();
        auto ix = xvar_.get_data().begin();
        auto iy = yvar_.get_data().begin();
        int n = 0;
        double xy = 0.;
        while (ix != xvar_.get_data().end()) {
            xy *= n;
            xy += (*(ix++) - xave_) * (*(iy++) - yave_);
            xy /= (++n);
        }
        rho_ = xy / (xrms_ * yrms_);
        INFO(name_ << ": n=" << n << ", <xy>=" << xy << ", rho=" << rho_);
        up2date_ = true;
    }

public:
    varstat2D(std::string xname, std::string yname) :
            name_(xname + yname), xvar_(xname), yvar_(yname),
            xave_(NAN), yave_(NAN), xrms_(NAN), yrms_(NAN), rho_(NAN), up2date_(true) {}

    void add_value(double xval, double yval) {
        xvar_.add_value(xval);
        yvar_.add_value(yval);
        up2date_ = false;
    }

    double averageX() {
        cache();
        return xave_;
    }

    double averageY() {
        cache();
        return yave_;
    }

    double rmsX() {
        cache();
        return xrms_;
    }

    double rmsY() {
        cache();
        return yrms_;
    }

    double rho() {
        cache();
        return rho_;
    }
};

}

#endif
