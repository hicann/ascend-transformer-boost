/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "lcal_sock_exchange.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <csignal>
#include <cerrno>
#include <cstring>
#include <set>
#include <string>
#include <fstream>
#include <sstream>

#include <sys/types.h>
#include <sys/socket.h>

#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#include <mki/utils/env/env.h>

using namespace std;
namespace Lcal {
const string LCAL_LOCAL_SOCK_IP = "127.0.0.1";
constexpr uint16_t LCAL_DEFAULT_SOCK_PORT = 10067;
constexpr uint32_t LCAL_MAX_BACK_LOG = 65536;

int ParseIpAndPort(const char* input, string &ip, uint16_t &port)
{
    if (input == nullptr) {
        return LCAL_INVALID_VALUE;
    }
    string inputStr(input);
    size_t colonPos = inputStr.find(":");
    if (colonPos == strings::npos) {
        MKI_LOG(ERROR) << "Input string does not contain a colon separating IP and port.";
        return LCAL_ERROR_INTERNAL;
    }

    ip = inputStr.substr(0, colonPos);
    std::string portStr = inputStr.substr(colonPos + 1);

    std::istringstream portStream(portStr);
    portStream >> port;
    if (portStream.fail() || portStream.bad()) {
        MKI_LOG(ERROR) << "Invalid port number.";
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}

LcalSockExchange::~LcalSockExchange()
{
    Cleanup();
}

LcalSockExchange::LcalSockExchange(int rank, int rankSize, std::vector<int> &rankList, int commDomain)
    : rank_(rank), rankSize_(rankSize), rankList_(rankList), commDomain_(commDomain)
{
}

LcalSockExchange::LcalSockExchange(int rank, int rankSize, LcalUniqueId lcalCommId)
    : rank_(rank), rankSize_(rankSize)
{
    lcalCommId_.uid = lcalCommId;
}

int LcalSockExchange::GetNodeNum()
{
    if (!isInit_ && Prepare() != LCAL_SUCCESS) {
        return LCAL_ERROR_INTERNAL;
    }
    isInit_ = true;
    const string filePath = "/proc/sys/kernel/rankdom/boot_id";
    ifstream fileStream(filePath);
    stringstream buffer;
    if (fileStream) {
        buffer << fileStream.rdbuf();
        fileStream.close();
    }
    const std::string uuid = buffer.str();
    MKI_LOG(DEBUG) << "rank:" << rank_ << " UUID " << uuid;

    set<string> uuidSet {};
    uuidSet.insert(uuid);
    int nodeNum = -1;
    if (IsServer()) {
        for (int i = 1; i < rankSize_; ++i) {
            if (Recv(clientFds_[i], const_cast<__caddr_t>(uuid.data()), uuid.size(), 0) <= 0) {
                MKI_LOG(ERROR) << "Server side recv rank " << i << " buffer failed";
                return LCAL_ERROR_INTERNAL;
            }
            uuidSet.insert(uuid);
        }
        nodeNum = static_cast<int>(uuidSet.size());
        for (int i = 1; i < rankSize_; ++i) {
            if (Send(clientFds_[i], &nodeNum, sizeof(int), 0) <= 0) {
                MKI_LOG(ERROR) << "Server side send rank " << i << " buffer failed";
                return LCAL_ERROR_INTERNAL;
            }
        }
    } else {
        if (Send(fd_, uuid.data(), uuid.size(), 0) <= 0) {
            MKI_LOG(ERROR) << "Client side " << rank_ << " send buffer failed";
            return LCAL_ERROR_INTERNAL;
        }
        if (Send(fd_, &nodeNum, sizeof(int), 0) <= 0) {
            MKI_LOG(ERROR) << "Client side " << rank_ << " recv buffer failed";
            return LCAL_ERROR_INTERNAL;
        }
    }
    return nodeNum;
}

void LcalSockExchange::GetIpAndPort()
{
    
}

}