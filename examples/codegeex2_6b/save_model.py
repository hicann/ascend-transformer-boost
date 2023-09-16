from transformers import AutoModel

model = AutoModel.from_pretrained("/mnt/vepfs/qinkai/release/codegeex2-6b/", trust_remote_code=True).cuda()
model.save_pretrained("./", max_shard_size="2000MB")