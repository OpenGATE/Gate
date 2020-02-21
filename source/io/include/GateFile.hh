//
// Created by mdupont on 17/12/18.
//

#pragma once

#include <iostream>
#include <string>

class GateFile
{

public:
  virtual bool is_open() = 0;
  virtual void close() = 0;
  const std::string &path() const;

protected:
  void open(const std::string& path, std::ios_base::openmode) ;
  std::ios_base::openmode m_mode;
  std::string m_path;
};

