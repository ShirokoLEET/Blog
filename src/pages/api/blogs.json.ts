import { getSortedPosts } from "@utils/content-utils";

export async function GET() {
	const posts = await getSortedPosts();
	const blogs = posts.map((post) => ({
		title: post.data.title,
		description: post.data.description,
	}));

	return new Response(
		JSON.stringify({
			total: blogs.length,
			blogs,
		}),
		{
			headers: {
				"Content-Type": "application/json; charset=utf-8",
			},
		},
	);
}
