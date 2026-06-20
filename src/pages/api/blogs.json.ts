import { getSortedPosts } from "@utils/content-utils";
import MarkdownIt from "markdown-it";
import sanitizeHtml from "sanitize-html";

const parser = new MarkdownIt();

function getExcerpt(body: string | undefined): string {
	const content = typeof body === "string" ? body : String(body || "");
	const text = sanitizeHtml(parser.render(content), {
		allowedAttributes: {},
		allowedTags: [],
	})
		.replace(/\s+/g, " ")
		.trim();

	return Array.from(text).slice(0, 50).join("");
}

export async function GET() {
	const posts = await getSortedPosts();
	const blogs = posts.map((post) => ({
		title: post.data.title,
		excerpt: getExcerpt(post.body),
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
