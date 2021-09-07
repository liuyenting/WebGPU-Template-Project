add_executable(
    mp0
    main.cu
)

target_link_libraries(
    mp0
    PRIVATE
        libwb
)
