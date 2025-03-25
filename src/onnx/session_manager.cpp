//
// Created by Kibae Shin on 2023/09/01.
//

#include "../onnxruntime_server.hpp"
#include <dlfcn.h> // For dlopen and dlerror
#include <iostream>

using namespace onnxruntime_server::onnx;

session_manager::session_manager(const model_bin_getter_t &model_bin_getter, long num_threads)
	: model_bin_getter(model_bin_getter), thread_pool(num_threads) {
	assert(model_bin_getter != nullptr);

	// Explicitly load custom operator library at startup
	const char* custom_ops_path = std::getenv("ORT_CUSTOM_OPS_LIB_PATH");
	if (custom_ops_path != nullptr) {
		// Load the library globally and immediately resolve symbols
		void* handle = dlopen(custom_ops_path, RTLD_NOW | RTLD_GLOBAL);
		if (!handle) {
			std::cerr << "Failed to load custom operator library ("
			          << custom_ops_path << "): " << dlerror() << std::endl;
			throw std::runtime_error(dlerror());
		} else {
			std::cout << "Successfully loaded custom operator library: " << custom_ops_path << std::endl;
		}
	} else {
		std::cerr << "Environment variable ORT_CUSTOM_OPS_LIB_PATH is not set!" << std::endl;
	}
}

session_manager::~session_manager() {
	thread_pool.flush();
}

session_ptr session_manager::get_session(const std::string &model_name, const std::string &model_version) {
	const auto key = session_key(model_name, model_version);
	return get_session(key);
}

session_ptr session_manager::get_session(const session_key &key) {
	std::lock_guard<std::recursive_mutex> lock(mutex);
	auto it = sessions.find(key);
	if (it == sessions.end())
		return nullptr;
	return it->second;
}

session_ptr session_manager::create_session(
	const std::string &model_name, const std::string &model_version, const json &option, const char *model_data,
	size_t model_data_length
) {
	auto key = session_key(model_name, model_version);

	session_ptr session = nullptr;
	std::lock_guard<std::recursive_mutex> lock(mutex);

	// Check if session already exists
	const auto current_session = get_session(key);
	if (current_session != nullptr)
		throw conflict_error("session already exists");

	// Create session based on provided data or file path
	if (model_data != nullptr && model_data_length > 0) {
		session = std::make_shared<onnx::session>(key, model_data, model_data_length, option);
	} else if (option.contains("path") && option["path"].is_string()) {
		session = std::make_shared<onnx::session>(key, option["path"].get<std::string>(), option);
	} else {
		auto model_bin = model_bin_getter(model_name, model_version);
		model_data = model_bin.data();
		model_data_length = model_bin.size();
		session = std::make_shared<onnx::session>(key, model_data, model_data_length, option);
	}
	sessions.emplace(key, session);
	return session;
}

void session_manager::remove_session(const std::string &model_name, const std::string &model_version) {
	auto key = session_key(model_name, model_version);
	remove_session(key);
}

void session_manager::remove_session(const session_key &key) {
	std::lock_guard<std::recursive_mutex> lock(mutex);
	auto it = sessions.find(key);
	if (it == sessions.end()) {
		throw not_found_error("session not found");
	}
	sessions.erase(it);
}
