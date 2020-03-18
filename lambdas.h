//
// Created by zhelenskiy on 18.03.2020.
//

#ifndef BUFFERED_TEMPLATE_LAMBDAS_H
#define BUFFERED_TEMPLATE_LAMBDAS_H
#define fn0(expr) [=]{return expr;}
#define fn1(expr) [=](const auto& item){return expr;}
#define fn2(expr) [=](const auto& first, const auto& second){return expr;}
#endif //BUFFERED_TEMPLATE_LAMBDAS_H
