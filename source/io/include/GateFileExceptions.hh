//
// Created by mdupont on 04/04/19.
//

#pragma once

#include <stdexcept>

class GateFileException : public std::runtime_error {
 public:
  explicit GateFileException(const std::string &arg);

};


class GateClosedFileException: public GateFileException {
 public:
  explicit GateClosedFileException(const std::string &arg);

};

class GateModeFileException: public GateFileException {
 public:
  explicit GateModeFileException(const std::string &arg);
};

class GateHeaderException: public GateFileException {
 public:
  explicit GateHeaderException(const std::string &arg);

};

class GateMissingHeaderException: public GateHeaderException {
 public:
  explicit GateMissingHeaderException(const std::string &arg);

};

class GateMalFormedHeaderException: public GateHeaderException {
 public:
  explicit GateMalFormedHeaderException(const std::string &arg);

};


class GateKeyAlreadyExistsException: public GateHeaderException {
 public:
  explicit GateKeyAlreadyExistsException(const std::string &arg);

};

class GateKeyNotFoundInHeaderException: public GateHeaderException {
public:
  explicit GateKeyNotFoundInHeaderException(const std::string &arg);
};

class GateTypeMismatchHeaderException: public GateHeaderException {
public:
  explicit GateTypeMismatchHeaderException(const std::string &arg);
};


class GateNoTypeInHeaderException: public GateHeaderException {
public:
  explicit GateNoTypeInHeaderException(const std::string &arg);

};



class GateManagerException: public GateFileException {
public:
  explicit GateManagerException(const std::string &arg);
};

class GateUnknownKindManagerException: public GateManagerException {
public:
  explicit GateUnknownKindManagerException(const std::string &arg);
};



